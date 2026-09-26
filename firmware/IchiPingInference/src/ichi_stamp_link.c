#include "ichi_stamp_link.h"

#include "irq.h"
#include "mcu.h"
#include "rdwr_reg.h"
#include "wdt.h"

/* I2F0CTL bits */
#define CTL_TXAK   (1UL << 3)
#define CTL_MTX    (1UL << 4)
#define CTL_MSTA   (1UL << 5)
/* I2F0SR bits */
#define SR_RXAK    (1UL << 0)
#define SR_MIF     (1UL << 1)
#define SR_MBB     (1UL << 5)
#define SR_MCF     (1UL << 7)

#define POLL_LIMIT     (200000UL)
#define REPLY_DELAY_US (400UL)       /* Stamp prepares the reply after the write */
#define RETRY_LIMIT    (8U)
#define CMD_INFO       (0x01U)
#define CMD_CLIP       (0x02U)
#define CMD_STATUS     (0x03U)
#define CMD_MEASURE    (0x20U)
#define CMD_MSTAT      (0x21U)
#define CMD_READ       (0x10U)

static uint8_t current_clip;
static uint32_t retry_count;
static uint32_t byte_count;
static uint8_t reply[4U + 2U * ICHI_STAMP_READ_MAX];

static void delay_us(uint32_t us)
{
    /* ~48 MHz, a few cycles per iteration; only needs to be "at least". */
    volatile uint32_t n = us * 12UL;
    while (n-- != 0UL) { }
}

static bool wait_sr(uint32_t mask, bool set)
{
    uint32_t n;
    for (n = 0UL; n < POLL_LIMIT; n++)
    {
        if (((I2CF0->I2F0SR & mask) != 0UL) == set)
        {
            return true;
        }
    }
    return false;
}

static void clear_transfer_flags(void)
{
    clear_bit(I2CF0->I2F0SR, SR_MCF | SR_MIF);
}

static void begin(void)
{
    irq_i2cf0_dis();
}

static void end(void)
{
    clear_transfer_flags();
    irq_i2cf0_clearIRQ();
    irq_i2cf0_ena();
}

static void stop(void)
{
    clear_bit(I2CF0->I2F0CTL, CTL_MSTA);
    (void)wait_sr(SR_MBB, false);
}

static bool write_bytes(uint8_t address, const uint8_t *data, uint8_t size)
{
    uint8_t i;
    bool ok = false;

    begin();
    if (wait_sr(SR_MBB, false))
    {
        set_bit(I2CF0->I2F0CTL, CTL_MTX);
        clear_bit(I2CF0->I2F0CTL, CTL_TXAK);
        write_reg32(I2CF0->I2F0DR, (uint32_t)(address << 1));
        set_bit(I2CF0->I2F0CTL, CTL_MSTA);
        ok = true;
        for (i = 0U; ok && (i <= size); i++)
        {
            ok = wait_sr(SR_MCF, true);
            clear_transfer_flags();
            if (ok && ((I2CF0->I2F0SR & SR_RXAK) != 0UL))
            {
                ok = false;                              /* NACK */
            }
            if (ok && (i < size))
            {
                write_reg32(I2CF0->I2F0DR, data[i]);
            }
        }
        stop();
    }
    end();
    return ok;
}

static bool read_bytes(uint8_t *data, uint16_t size)
{
    uint16_t i;
    bool ok = false;

    begin();
    if (wait_sr(SR_MBB, false))
    {
        clear_bit(I2CF0->I2F0CTL, CTL_MTX);              /* master receive */
        if (size == 1U) { set_bit(I2CF0->I2F0CTL, CTL_TXAK); }
        else { clear_bit(I2CF0->I2F0CTL, CTL_TXAK); }
        write_reg32(I2CF0->I2F0DR, (uint32_t)((ICHI_STAMP_ADDR << 1) | 1U));
        set_bit(I2CF0->I2F0CTL, CTL_MSTA);
        ok = wait_sr(SR_MCF, true);
        if (ok && ((I2CF0->I2F0SR & SR_RXAK) != 0UL))
        {
            ok = false;
        }
        clear_transfer_flags();
        if (ok)
        {
            (void)I2CF0->I2F0DR;                         /* dummy read starts byte 0 */
            for (i = 0U; ok && (i < size); i++)
            {
                ok = wait_sr(SR_MCF, true);
                if (i + 2U == size) { set_bit(I2CF0->I2F0CTL, CTL_TXAK); }     /* NACK the last */
                if (i + 1U == size)
                {
                    clear_bit(I2CF0->I2F0CTL, CTL_MSTA);                     /* STOP */
                    clear_bit(I2CF0->I2F0CTL, CTL_TXAK);
                }
                data[i] = (uint8_t)I2CF0->I2F0DR;        /* also starts the next byte */
                clear_transfer_flags();
            }
        }
        if (!ok) { clear_bit(I2CF0->I2F0CTL, CTL_MSTA); }
        (void)wait_sr(SR_MBB, false);
        clear_bit(I2CF0->I2F0CTL, CTL_TXAK);
    }
    end();
    if (ok) { byte_count += size; }
    return ok;
}

/* Write a command, then read reply_size bytes whose first check_size bytes
   must equal expect (stale-reply detection).  Retries with a longer delay. */
static bool exchange(const uint8_t *cmd, uint8_t cmd_size, uint16_t reply_size,
                     const uint8_t *expect, uint8_t check_size)
{
    uint8_t attempt;
    for (attempt = 0U; attempt < RETRY_LIMIT; attempt++)
    {
        uint8_t i;
        bool match = true;
        if (attempt != 0U) { retry_count++; }
        wdt_clear();
        if (!write_bytes(ICHI_STAMP_ADDR, cmd, cmd_size))
        {
            delay_us(2000UL);
            continue;
        }
        delay_us(REPLY_DELAY_US * (1UL + attempt));
        if (!read_bytes(reply, reply_size))
        {
            continue;
        }
        for (i = 0U; i < check_size; i++)
        {
            if (reply[i] != expect[i]) { match = false; }
        }
        if (match) { return true; }
    }
    return false;
}

bool IchiStampGetInfo(IchiStampInfo *info)
{
    static const uint8_t cmd[1] = {CMD_INFO};
    static const uint8_t expect[2] = {'I', 'C'};
    if (!exchange(cmd, 1U, 8U, expect, 2U)) { return false; }
    info->clips = reply[2];
    info->baseline = reply[3];
    info->samples = (uint16_t)(reply[4] | ((uint16_t)reply[5] << 8));
    info->version = reply[6];
    return true;
}

bool IchiStampGetClip(uint8_t clip, uint8_t *class_id)
{
    uint8_t cmd[2] = {CMD_CLIP, clip};
    uint8_t expect[1] = {clip};
    if (!exchange(cmd, 2U, 4U, expect, 1U)) { return false; }
    *class_id = reply[1];
    return true;
}

bool IchiStampGetSwitches(uint8_t *state, bool *exec_pressed)
{
    static const uint8_t cmd[1] = {CMD_STATUS};
    static const uint8_t expect[1] = {'S'};
    if (!exchange(cmd, 1U, 4U, expect, 1U)) { return false; }
    *state = (uint8_t)(reply[1] & 0x1FU);
    *exec_pressed = (reply[2] & 1U) != 0U;
    return true;
}

bool IchiStampMeasure(uint8_t mode)
{
    const uint8_t start[2] = {CMD_MEASURE, mode};
    static const uint8_t poll[1] = {CMD_MSTAT};
    static const uint8_t expect[1] = {'M'};
    uint32_t n;
    uint8_t id;
    if (!exchange(start, 2U, 4U, expect, 1U)) { return false; }
    id = reply[2];
    for (n = 0UL; n < 1500UL; n++)                      /* 1500 x 10 ms = 15 s */
    {
        delay_us(10000UL);
        wdt_clear();
        if (!exchange(poll, 1U, 4U, expect, 1U)) { continue; }
        if (reply[2] != id) { return false; }
        if (reply[1] == 2U) { return true; }            /* done */
        if (reply[1] == 3U) { return false; }           /* capture / alignment failed */
    }
    return false;
}

void IchiStampSelectClip(uint8_t clip)
{
    current_clip = clip;
}

bool IchiStampReadPcm(uint16_t offset, int16_t *dst, uint16_t count)
{
    while (count > 0U)
    {
        uint8_t n = (uint8_t)((count > ICHI_STAMP_READ_MAX) ? ICHI_STAMP_READ_MAX : count);
        uint8_t cmd[5] = {CMD_READ, current_clip, (uint8_t)offset, (uint8_t)(offset >> 8), n};
        uint8_t expect[4] = {(uint8_t)offset, (uint8_t)(offset >> 8), current_clip, n};
        uint8_t i;
        if (!exchange(cmd, 5U, (uint16_t)(4U + 2U * n), expect, 4U))
        {
            return false;
        }
        for (i = 0U; i < n; i++)
        {
            dst[i] = (int16_t)(uint16_t)(reply[4U + 2U * i] | ((uint16_t)reply[5U + 2U * i] << 8));
        }
        dst += n;
        offset = (uint16_t)(offset + n);
        count = (uint16_t)(count - n);
    }
    return true;
}

bool IchiI2cWrite(uint8_t address, const uint8_t *data, uint8_t size)
{
    return write_bytes(address, data, size);
}

uint32_t IchiStampRetryCount(void)
{
    return retry_count;
}

uint32_t IchiStampByteCount(void)
{
    return byte_count;
}

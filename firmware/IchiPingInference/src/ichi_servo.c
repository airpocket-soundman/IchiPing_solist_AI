#include "ichi_servo.h"

#include "ichi_stamp_link.h"
#include "mcu.h"
#include "rdwr_reg.h"

#define PCA_ADDR       (0x40U)
#define REG_MODE1      (0x00U)
#define REG_MODE2      (0x01U)
#define REG_LED0       (0x06U)          /* LEDn_ON_L at 0x06 + 4n */
#define REG_ALL_LED    (0xFAU)
#define REG_PRESCALE   (0xFEU)
#define MODE1_AI       (0x20U)          /* register auto-increment */
#define MODE1_SLEEP    (0x10U)
#define MODE2_OUTDRV   (0x04U)          /* totem-pole outputs */
#define PRESCALE_50HZ  (121U)           /* 25 MHz / (4096 x 122) = 50.0 Hz */
#define PERIOD_US      (20000UL)
#define OE_BIT         (1UL << 2)       /* P42 */
#define MODE_OUTPUT    (0x02UL)

static bool write_reg(uint8_t reg, uint8_t value)
{
    uint8_t data[2] = { reg, value };
    return IchiI2cWrite(PCA_ADDR, data, 2U);
}

static void wait_short(void)
{
    volatile uint32_t n = 48000UL / 4UL;   /* > 500 us oscillator start-up */
    while (n-- > 0U) { }
}

void IchiServoEnable(bool enable)
{
    if (enable) { PORT4->P4DO &= ~OE_BIT; } else { PORT4->P4DO |= OE_BIT; }
}

bool IchiServoInitialize(void)
{
    static const uint8_t all_off[5] = { REG_ALL_LED, 0U, 0U, 0U, 0x10U };   /* ALL_LED_OFF_H full-off */
    bool ok;

    PORT4->P4DO |= OE_BIT;                                          /* outputs off first */
    write_bit(PORT4->P4MOD0, (0xFFUL << 16), (MODE_OUTPUT << 16));
    ok = write_reg(REG_MODE1, MODE1_SLEEP | MODE1_AI);               /* prescaler is writable only in sleep */
    ok = ok && write_reg(REG_PRESCALE, PRESCALE_50HZ);
    ok = ok && write_reg(REG_MODE2, MODE2_OUTDRV);
    ok = ok && IchiI2cWrite(PCA_ADDR, all_off, sizeof(all_off));
    ok = ok && write_reg(REG_MODE1, MODE1_AI);                       /* wake */
    wait_short();
    return ok;
}

bool IchiServoSetCount(uint8_t channel, uint16_t counts)
{
    uint8_t data[5];
    if ((channel >= 16U) || (counts > 4095U)) { return false; }
    data[0] = (uint8_t)(REG_LED0 + 4U * channel);
    data[1] = 0U;                                                    /* ON at count 0 */
    data[2] = 0U;
    data[3] = (uint8_t)counts;                                       /* OFF at pulse end */
    data[4] = (uint8_t)(counts >> 8);
    return IchiI2cWrite(PCA_ADDR, data, 5U);
}

bool IchiServoSetPulse(uint8_t channel, uint16_t pulse_us)
{
    if (pulse_us < 1000U) { pulse_us = 1000U; }
    if (pulse_us > 2000U) { pulse_us = 2000U; }
    return IchiServoSetCount(channel, (uint16_t)(((uint32_t)pulse_us * 4096UL + PERIOD_US / 2UL) / PERIOD_US));
}

bool IchiServoRelease(uint8_t channel)
{
    uint8_t data[5];
    if (channel >= 16U) { return false; }
    data[0] = (uint8_t)(REG_LED0 + 4U * channel);
    data[1] = 0U;
    data[2] = 0U;
    data[3] = 0U;
    data[4] = 0x10U;                                                 /* LEDn_OFF_H full-off */
    return IchiI2cWrite(PCA_ADDR, data, 5U);
}

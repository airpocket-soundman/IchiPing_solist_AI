/*
 * Solist-AI side of the Solist <-> Stamp-S3A I2C link test (DT-EBML63Q2557).
 *
 * Replaces S_System/main.c in a disposable copy of the private LEXIDE project.
 * The vendor I2CF0 master driver is write-only, so this test checks writes and
 * the slave ACK/NACK; reads come in a later step.
 *
 *   SW2: send "PINGnnnn" to 0x42 once
 *   SW3: toggle auto ping (every 200 ms)
 *   SW4: clear counters
 *   SW5: scan 0x08..0x77 and list the ACKing addresses
 *
 *   LED (LED1 LED2 LED3): 010 = ACK, 101 = NACK, 111 = timeout
 */
#include <stdbool.h>
#include <stdint.h>

#include "Lcd.h"
#include "LcdI2cf0.h"
#include "PeriodicHandler10ms.h"
#include "Regulator5VOutput.h"
#include "Sleep.h"
#include "SoftwareInterrupt.h"
#include "SystemError.h"
#include "SystemPowerControl.h"
#include "Sw.h"
#include "TimeControl.h"
#include "clock.h"
#include "irq.h"
#include "mcu.h"
#include "smpl_common.h"
#include "smpl_common_led.h"
#include "timer0_1.h"
#include "wdt.h"

#define STAMP_ADDR_7BIT   (0x42U)
#define SW_MASK           (0x0FU)  /* P50..P53 = SW2..SW5 */
#define DEBOUNCE_TICKS    (3U)
#define XFER_TIMEOUT_TICK (5U)     /* 50 ms */
#define AUTO_PERIOD_TICK  (20U)    /* 200 ms */
#define LCD_COLUMNS       (LCD_MOST_CHARACTERS_ON_A_LINE)

typedef enum { XFER_ACK, XFER_NACK, XFER_TIMEOUT } XferResult;

static volatile uint8_t pressed_events;
static volatile uint32_t tick_10ms;
static uint8_t stable_level = SW_MASK;
static uint8_t last_raw = SW_MASK;
static uint8_t same_count;

static volatile bool xfer_done;
static volatile uint8_t xfer_err;

static void periodic_10ms(void)
{
    uint8_t raw = (uint8_t)(PORT5->P5DI & SW_MASK);

    tick_10ms++;
    if (raw != last_raw)
    {
        last_raw = raw;
        same_count = 0U;
        return;
    }
    if (same_count < DEBOUNCE_TICKS)
    {
        same_count++;
        if (same_count == DEBOUNCE_TICKS)
        {
            pressed_events |= (uint8_t)(stable_level & (uint8_t)~raw);
            stable_level = raw;
        }
    }
}

static void xfer_callback(uint32_t size, uint8_t err_stat)
{
    (void)size;
    xfer_err = err_stat;
    xfer_done = true;
}

/* Blocking write through the vendor I2CF0 driver (address is 7-bit). */
static XferResult i2c_write(uint8_t addr7, uint8_t *data, uint16_t size)
{
    uint32_t start;

    xfer_done = false;
    xfer_err = 0U;
    (void)LcdI2cf0Write((uint8_t)(addr7 << 1), data, size, xfer_callback);
    start = tick_10ms;
    while (!xfer_done)
    {
        if ((tick_10ms - start) > XFER_TIMEOUT_TICK)
        {
            return XFER_TIMEOUT;
        }
    }
    return (xfer_err == 0U) ? XFER_ACK : XFER_NACK;
}

static void show_leds(uint8_t value)
{
    if ((value & 0x04U) != 0U) { smpl_onLED1(); } else { smpl_offLED1(); }
    if ((value & 0x02U) != 0U) { smpl_onLED2(); } else { smpl_offLED2(); }
    if ((value & 0x01U) != 0U) { smpl_onLED3(); } else { smpl_offLED3(); }
}

static bool draw_line(uint8_t position, const char *text)
{
    char line[LCD_COLUMNS + 1U];
    uint8_t i = 0U;

    while ((i < LCD_COLUMNS) && (text[i] != '\0'))
    {
        line[i] = text[i];
        i++;
    }
    while (i < LCD_COLUMNS)
    {
        line[i++] = ' ';
    }
    line[LCD_COLUMNS] = '\0';
    return LcdDraw(position, line);
}

static void put_dec4(char *dst, uint16_t v)
{
    dst[0] = (char)('0' + ((v / 1000U) % 10U));
    dst[1] = (char)('0' + ((v / 100U) % 10U));
    dst[2] = (char)('0' + ((v / 10U) % 10U));
    dst[3] = (char)('0' + (v % 10U));
}

static void put_hex2(char *dst, uint8_t v)
{
    static const char hex[] = "0123456789ABCDEF";
    dst[0] = hex[(v >> 4) & 0x0FU];
    dst[1] = hex[v & 0x0FU];
}

static uint16_t ok_count;
static uint16_t ng_count;
static uint16_t seq;

static void send_ping(void)
{
    static const char *const result_text[3] = {"ACK", "NACK", "TIMEOUT"};
    uint8_t frame[8] = {'P', 'I', 'N', 'G', '0', '0', '0', '0'};
    char line1[LCD_COLUMNS + 1U] = "PING0000 ";
    char line2[LCD_COLUMNS + 1U] = "OK0000 NG0000";
    XferResult r;
    uint8_t i;

    seq++;
    put_dec4((char *)&frame[4], seq);
    r = i2c_write(STAMP_ADDR_7BIT, frame, sizeof(frame));
    if (r == XFER_ACK) { ok_count++; } else { ng_count++; }

    put_dec4(&line1[4], seq);
    for (i = 0U; (result_text[r][i] != '\0') && ((9U + i) < LCD_COLUMNS); i++)
    {
        line1[9U + i] = result_text[r][i];
    }
    line1[9U + i] = '\0';
    put_dec4(&line2[2], ok_count);
    put_dec4(&line2[9], ng_count);
    (void)draw_line(LCD_START_OF_FIRST_LINE, line1);
    (void)draw_line(LCD_START_OF_SECOND_LINE, line2);
    show_leds((r == XFER_ACK) ? 0x02U : ((r == XFER_NACK) ? 0x05U : 0x07U));
}

/* Probe every 7-bit address with a single 0x00 byte (an empty LCD control
   byte is harmless).  Lists up to four ACKing addresses. */
static void scan_bus(void)
{
    char line2[LCD_COLUMNS + 1U] = "";
    uint8_t found = 0U;
    uint8_t addr;
    uint8_t pos = 0U;

    (void)draw_line(LCD_START_OF_FIRST_LINE, "Scanning...");
    for (addr = 0x08U; addr <= 0x77U; addr++)
    {
        uint8_t probe = 0x00U;
        wdt_clear();
        if (i2c_write(addr, &probe, 1U) == XFER_ACK)
        {
            found++;
            if (pos <= (LCD_COLUMNS - 3U))
            {
                put_hex2(&line2[pos], addr);
                line2[pos + 2U] = ' ';
                pos = (uint8_t)(pos + 3U);
            }
        }
    }
    line2[pos] = '\0';
    {
        char line1[LCD_COLUMNS + 1U] = "Scan found 00";
        line1[11] = (char)('0' + ((found / 10U) % 10U));
        line1[12] = (char)('0' + (found % 10U));
        (void)draw_line(LCD_START_OF_FIRST_LINE, line1);
    }
    (void)draw_line(LCD_START_OF_SECOND_LINE, (found != 0U) ? line2 : "(none)");
    show_leds(found);
}

int32_t main(void)
{
    bool auto_ping = false;
    uint32_t last_auto = 0U;

    __disable_irq();
    wdt_init(WDT_2S);
    wdt_clear();
    smpl_setLsCrystal32Khz();
    smpl_setHsPll48Mhz(CLK_XSPEN_DIS, CLK_HXSPEN_DIS);
    __enable_irq();

    SystemPowerControlInit();
    SoftwareInterruptInit();
    TimeControlInit();
    PeriodicHandler10msInit();
    timer0_start();
    SystemErrorInit();

    smpl_initLED1(LED_ACTIVE);
    smpl_initLED2(LED_ACTIVE);
    smpl_initLED3(LED_ACTIVE);
    show_leds(0x07U);

    SwInit();
    Regulator5VOutputInit();
    Regulator5VOutputOn();
    LcdPeripheralInit();
    LcdInit();
    LcdDisplayOnOff(LCD_DISPLAY_ON, LCD_CURSOR_OFF, LCD_CURSOR_BLINK_OFF);
    LcdClearDisplay();
    (void)draw_line(LCD_START_OF_FIRST_LINE, "Stamp link test");
    (void)draw_line(LCD_START_OF_SECOND_LINE, "2:PING 3:AUTO 5:S");
    LcdBacklightOn();
    show_leds(0x00U);

    PeriodicHandler10msSetCallBack(periodic_10ms);

    for (;;)
    {
        uint8_t events;

        __disable_irq();
        events = pressed_events;
        pressed_events = 0U;
        __enable_irq();

        if ((events & 0x01U) != 0U) { send_ping(); }
        if ((events & 0x02U) != 0U)
        {
            auto_ping = !auto_ping;
            last_auto = tick_10ms;
            (void)draw_line(LCD_START_OF_FIRST_LINE, auto_ping ? "AUTO PING on" : "AUTO PING off");
        }
        if ((events & 0x04U) != 0U)
        {
            ok_count = 0U;
            ng_count = 0U;
            seq = 0U;
            (void)draw_line(LCD_START_OF_FIRST_LINE, "Counters cleared");
            (void)draw_line(LCD_START_OF_SECOND_LINE, "OK0000 NG0000");
            show_leds(0x00U);
        }
        if ((events & 0x08U) != 0U)
        {
            auto_ping = false;
            scan_bus();
        }
        if (auto_ping && ((tick_10ms - last_auto) >= AUTO_PERIOD_TICK))
        {
            last_auto = tick_10ms;
            send_ping();
        }
        wdt_clear();
        if (!auto_ping)
        {
            SleepChangetoHaltMode();
        }
    }
}

/*
 * Servo bring-up (tools/build.ps1 -Main ichi_servo_test_main).
 * Replaces S_System/main.c of the vendor project.
 *
 * Safe sequence: all PCA9685 outputs off and OE High at start; nothing moves until EXEC
 * is pressed.  Then ch0..ch4 (a, b, c, AB, BC) are moved one at a time
 * 1500 -> 1300 -> 1700 -> 1500 us (about +-18 degrees) and released.
 * Pressing EXEC again while moving stops everything at once (OE High, all outputs off).
 * EXEC comes from the Stamp over I2C (firmware/StampMicTest STATUS 0x03).
 */
#include <stdbool.h>
#include <stdint.h>

#include "Lcd.h"
#include "PeriodicHandler10ms.h"
#include "Regulator5VOutput.h"
#include "SoftwareInterrupt.h"
#include "SystemError.h"
#include "SystemPowerControl.h"
#include "TimeControl.h"
#include "clock.h"
#include "ichi_servo.h"
#include "ichi_stamp_link.h"
#include "ichi_tft.h"
#include "mcu.h"
#include "smpl_common.h"
#include "timer0_1.h"
#include "wdt.h"

static volatile uint32_t tick_10ms;
static bool exec_now;
static bool abort_request;

static void periodic_10ms(void)
{
    tick_10ms++;
}

static void put_number(char *dst, uint32_t value, uint8_t digits)
{
    while (digits-- > 0U)
    {
        dst[digits] = (char)('0' + value % 10U);
        value /= 10U;
    }
}

/* Copies src into dst, padding with spaces to width characters. */
static void copy_padded(char *dst, const char *src, uint8_t width)
{
    uint8_t k;
    for (k = 0U; k < width; k++)
    {
        dst[k] = (*src != '\0') ? *src++ : ' ';
    }
}

static void status(const char *tft_text, uint16_t color, const char *lcd_text)
{
    IchiTftFillRect(0U, 60U, ICHI_TFT_WIDTH, 40U, ICHI_BLACK);
    IchiTftDrawText(8U, 64U, tft_text, color, ICHI_BLACK, 3U);
    (void)LcdDraw(LCD_START_OF_FIRST_LINE, lcd_text);
}

/* Returns true on a new EXEC press (polls the Stamp every 10 ms tick). */
static bool exec_pressed_edge(void)
{
    static uint32_t last = 0U;
    uint8_t state;
    bool pressed;
    bool edge = false;
    if ((tick_10ms != last) && IchiStampGetSwitches(&state, &pressed))
    {
        last = tick_10ms;
        edge = pressed && !exec_now;
        exec_now = pressed;
    }
    wdt_clear();
    return edge;
}

/* Waits n x 10 ms; an EXEC press requests an abort. */
static void wait_ticks(uint32_t n)
{
    uint32_t start = tick_10ms;
    while ((tick_10ms - start) < n)
    {
        if (exec_pressed_edge()) { abort_request = true; }
        if (abort_request) { return; }
    }
}

static void stop_all(void)
{
    uint8_t ch;
    IchiServoEnable(false);
    for (ch = 0U; ch < ICHI_SERVO_CHANNELS; ch++) { (void)IchiServoRelease(ch); }
}

int32_t main(void)
{
    static const char *const names[ICHI_SERVO_CHANNELS] = { "WIN A", "WIN B", "WIN C", "DOOR AB", "DOOR BC" };
    static const uint16_t steps[4] = { 1500U, 1300U, 1700U, 1500U };
    bool pca_ok;

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
    PeriodicHandler10msSetCallBack(periodic_10ms);
    timer0_start();
    SystemErrorInit();

    Regulator5VOutputInit();
    Regulator5VOutputOn();
    LcdPeripheralInit();
    LcdInit();
    LcdDisplayOnOff(LCD_DISPLAY_ON, LCD_CURSOR_OFF, LCD_CURSOR_BLINK_OFF);
    LcdClearDisplay();
    LcdBacklightOn();

    IchiTftInitialize();                         /* also drives P42 (PCA OE) High */
    IchiTftFillRect(0U, 0U, ICHI_TFT_WIDTH, ICHI_TFT_HEIGHT, ICHI_BLACK);
    IchiTftDrawText(8U, 10U, "SERVO TEST", ICHI_WHITE, ICHI_BLACK, 3U);
    IchiTftDrawText(8U, 200U, "EXEC: START / STOP ALL", ICHI_CYAN, ICHI_BLACK, 2U);

    pca_ok = IchiServoInitialize();
    if (!pca_ok)
    {
        status("PCA9685 NO ACK", ICHI_RED, "PCA9685 no ACK  ");
        for (;;) { wdt_clear(); }
    }

    for (;;)
    {
        uint8_t ch, i;
        status("PRESS EXEC", ICHI_YELLOW, "servo: press EXE");
        stop_all();
        while (!exec_pressed_edge()) { }
        abort_request = false;
        IchiServoEnable(true);
        for (ch = 0U; (ch < ICHI_SERVO_CHANNELS) && !abort_request; ch++)
        {
            for (i = 0U; (i < 4U) && !abort_request; i++)
            {
                char tft[] = "CH0 DOOR BC 0000US";
                char lcd[] = "ch0 0000us      ";
                tft[2] = (char)('0' + ch);
                lcd[2] = (char)('0' + ch);
                copy_padded(&tft[4], names[ch], 7U);
                put_number(&tft[12], steps[i], 4U);
                put_number(&lcd[4], steps[i], 4U);
                status(tft, ICHI_GREEN, lcd);
                if (!IchiServoSetPulse(ch, steps[i]))
                {
                    status("PCA9685 WRITE FAIL", ICHI_RED, "PCA write fail  ");
                    abort_request = true;
                    break;
                }
                wait_ticks((i == 0U) ? 150U : 100U);
            }
            (void)IchiServoRelease(ch);
        }
        stop_all();
        if (abort_request)
        {
            status("STOPPED BY EXEC", ICHI_RED, "stopped by EXEC ");
            wait_ticks(150U);
            abort_request = false;
        }
        else
        {
            status("DONE", ICHI_WHITE, "servo test done ");
            wait_ticks(150U);
        }
    }
}

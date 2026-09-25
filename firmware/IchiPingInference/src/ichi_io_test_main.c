/*
 * Switch / EXEC bring-up on the TFT (tools/build.ps1 -Main ichi_io_test_main).
 * Replaces S_System/main.c of the vendor project.
 *
 * The Stamp-S3A reads the five door/window switches and EXEC (firmware/StampMicTest)
 * and returns them over I2C (STATUS 0x03).  There is no Stamp->Solist interrupt line on the
 * interposer, so the Solist polls every 10 ms (the Stamp detects edges by GPIO interrupt) and redraws
 * only what changed.  The on-board character LCD mirrors the state.  PCA9685 OE is High.
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
#include "ichi_stamp_link.h"
#include "ichi_tft.h"
#include "mcu.h"
#include "smpl_common.h"
#include "timer0_1.h"
#include "wdt.h"

#define GRAY       ICHI_RGB565(64, 64, 64)
#define DARK_RED   ICHI_RGB565(170, 0, 0)
#define DARK_GREEN ICHI_RGB565(0, 150, 0)

static volatile uint32_t tick_10ms;

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

/* Screen order, left to right: C, BC, B, AB, A (state bits: a=0 b=1 c=2 AB=3 BC=4). */
static const uint8_t screen_bit[5] = { 2U, 4U, 1U, 3U, 0U };

static void draw_switch(uint8_t slot, bool open)
{
    static const char *const labels[5] = { "C", "BC", "B", "AB", "A" };
    uint16_t x = (uint16_t)(8U + slot * 62U);
    uint8_t index = slot;
    uint16_t color = open ? DARK_GREEN : DARK_RED;
    IchiTftFillRect(x, 52U, 58U, 56U, color);
    IchiTftDrawText((uint16_t)(x + (labels[index][1] ? 11U : 17U)), 58U, labels[index], ICHI_WHITE, color, 3U);
    IchiTftDrawText((uint16_t)(x + (open ? 17U : 14U)), 90U, open ? "OPEN" : "CLOSE", ICHI_WHITE, color, 1U);
}

static void draw_exec(bool pressed, uint32_t presses)
{
    char text[] = "EXEC 0000";
    uint16_t color = pressed ? ICHI_YELLOW : GRAY;
    put_number(&text[5], presses, 4U);
    IchiTftFillRect(8U, 120U, 304U, 40U, color);
    IchiTftDrawText(40U, 128U, text, pressed ? ICHI_BLACK : ICHI_WHITE, color, 3U);
}

static void draw_state(uint8_t state)
{
    char text[] = "S00000 (A B C AB BC)";
    uint8_t bit;
    for (bit = 0U; bit < 5U; bit++)
    {
        text[1U + bit] = (char)('0' + ((state >> bit) & 1U));
    }
    IchiTftDrawText(8U, 176U, text, ICHI_CYAN, ICHI_BLACK, 2U);
}

static void draw_link(bool ok, uint32_t good, uint32_t bad)
{
    char text[] = "LINK OK 000000 ERR 0000";
    if (!ok) { text[5] = 'N'; text[6] = 'G'; }
    put_number(&text[8], good, 6U);
    put_number(&text[19], bad, 4U);
    IchiTftDrawText(8U, 226U, text, ok ? ICHI_GREEN : ICHI_RED, ICHI_BLACK, 1U);
}

int32_t main(void)
{
    char lcd1[] = "IO s00000 EXEC0 ";
    char lcd2[] = "link ok 000000  ";
    uint8_t state = 0U, shown_state = 0xFFU;
    bool exec = false, shown_exec = true, link_ok = false, shown_link = true;
    uint32_t presses = 0U, good = 0U, bad = 0U, last_poll = 0U, last_link_draw = 0U;
    uint8_t i;

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
    LcdPeripheralInit();          /* also configures the shared I2CF0 bus used for the Stamp */
    LcdInit();
    LcdDisplayOnOff(LCD_DISPLAY_ON, LCD_CURSOR_OFF, LCD_CURSOR_BLINK_OFF);
    LcdClearDisplay();
    LcdBacklightOn();
    (void)LcdDraw(LCD_START_OF_FIRST_LINE, "IO test: init   ");

    IchiTftInitialize();
    IchiTftFillRect(0U, 0U, ICHI_TFT_WIDTH, ICHI_TFT_HEIGHT, ICHI_BLACK);
    IchiTftDrawText(8U, 10U, "SWITCH TEST", ICHI_WHITE, ICHI_BLACK, 3U);

    for (;;)
    {
        if (tick_10ms != last_poll)                  /* every 10 ms tick */
        {
            uint8_t s;
            bool e;
            last_poll = tick_10ms;
            link_ok = IchiStampGetSwitches(&s, &e);
            if (link_ok)
            {
                good++;
                if (e && !exec) { presses++; }
                state = s;
                exec = e;
            }
            else
            {
                bad++;
            }
        }
        if (state != shown_state)
        {
            for (i = 0U; i < 5U; i++)
            {
                uint8_t b = screen_bit[i];
                bool open = ((state >> b) & 1U) != 0U;
                if ((shown_state == 0xFFU) || (((shown_state >> b) & 1U) != (uint8_t)open))
                {
                    draw_switch(i, open);
                }
            }
            draw_state(state);
            shown_state = state;
        }
        if (exec != shown_exec)
        {
            draw_exec(exec, presses);
            shown_exec = exec;
        }
        if ((link_ok != shown_link) || ((tick_10ms - last_link_draw) >= 100U))
        {
            draw_link(link_ok, good, bad);
            shown_link = link_ok;
            last_link_draw = tick_10ms;
            for (i = 0U; i < 5U; i++)
            {
                lcd1[4U + i] = (char)('0' + ((state >> i) & 1U));
            }
            lcd1[14] = exec ? '1' : '0';
            lcd2[5] = link_ok ? 'o' : 'N';
            lcd2[6] = link_ok ? 'k' : 'G';
            put_number(&lcd2[8], good, 6U);
            (void)LcdDraw(LCD_START_OF_FIRST_LINE, lcd1);
            (void)LcdDraw(LCD_START_OF_SECOND_LINE, lcd2);
        }
        wdt_clear();
    }
}

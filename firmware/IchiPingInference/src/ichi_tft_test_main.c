/*
 * ILI9341 TFT bring-up test (tools/build.ps1 -Main ichi_tft_test_main).
 * Replaces S_System/main.c of the vendor project.
 *
 * Draws colour bars, a title and a counter that increments about once per second.
 * The on-board character LCD shows the progress, so a blank TFT can be told apart
 * from a firmware that is not running.  PCA9685 OE (P42) is held High.
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
#include "ichi_tft.h"
#include "mcu.h"
#include "smpl_common.h"
#include "timer0_1.h"
#include "wdt.h"

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

int32_t main(void)
{
    static const uint16_t bars[6] = { ICHI_RED, ICHI_GREEN, ICHI_BLUE, ICHI_WHITE, ICHI_YELLOW, ICHI_CYAN };
    char counter[] = "COUNT 0000";
    char lcd_line[] = "TFT count 0000  ";
    uint32_t count = 0U;
    uint32_t last = 0U;
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
    LcdPeripheralInit();
    LcdInit();
    LcdDisplayOnOff(LCD_DISPLAY_ON, LCD_CURSOR_OFF, LCD_CURSOR_BLINK_OFF);
    LcdClearDisplay();
    LcdBacklightOn();
    (void)LcdDraw(LCD_START_OF_FIRST_LINE, "TFT test: init  ");

    IchiTftInitialize();
    (void)LcdDraw(LCD_START_OF_FIRST_LINE, "TFT test: draw  ");
    IchiTftFillRect(0U, 0U, ICHI_TFT_WIDTH, ICHI_TFT_HEIGHT, ICHI_BLACK);
    for (i = 0U; i < 6U; i++)
    {
        IchiTftFillRect((uint16_t)(i * 53U), 0U, 53U, 60U, bars[i]);
    }
    IchiTftDrawText(16U, 84U, "ICHIPING TFT OK", ICHI_WHITE, ICHI_BLACK, 3U);
    IchiTftDrawText(16U, 124U, "SOLIST-AI ML63Q2557", ICHI_GREEN, ICHI_BLACK, 2U);
    IchiTftFillRect(0U, 236U, ICHI_TFT_WIDTH, 4U, ICHI_MAGENTA);   /* bottom edge marker */
    (void)LcdDraw(LCD_START_OF_FIRST_LINE, "TFT test: run   ");

    for (;;)
    {
        if ((tick_10ms - last) >= 100U)
        {
            last = tick_10ms;
            count++;
            put_number(&counter[6], count, 4U);
            put_number(&lcd_line[10], count, 4U);
            IchiTftDrawText(16U, 170U, counter, ICHI_YELLOW, ICHI_BLACK, 4U);
            (void)LcdDraw(LCD_START_OF_SECOND_LINE, lcd_line);
        }
        wdt_clear();
    }
}

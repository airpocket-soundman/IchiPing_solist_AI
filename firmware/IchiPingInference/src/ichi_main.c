/*
 * Replacement for S_System/main.c of the vendor AIVibrationInference project
 * (tools/build.ps1 copies it there).  Vendor sources are not vendored here.
 */
#include <stdbool.h>
#include <stdint.h>

#include "Lcd.h"
#include "PeriodicHandler10ms.h"
#include "Regulator5VOutput.h"
#include "Sleep.h"
#include "SoftwareInterrupt.h"
#include "SystemError.h"
#include "SystemPowerControl.h"
#include "TimeControl.h"
#include "Uart1.h"
#include "ichi_app.h"
#include "clock.h"
#include "mcu.h"
#include "smpl_common.h"
#include "timer0_1.h"
#include "wdt.h"

int32_t main(void)
{
    bool lcd_ready;

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
    Uart1PeripheralInit();
    SystemErrorInit();

    Regulator5VOutputInit();
    Regulator5VOutputOn();          /* LCD supply */
    LcdPeripheralInit();
    LcdInit();
    LcdDisplayOnOff(LCD_DISPLAY_ON, LCD_CURSOR_OFF, LCD_CURSOR_BLINK_OFF);
    LcdClearDisplay();
    lcd_ready = LcdDraw(LCD_START_OF_FIRST_LINE, "IchiPing ready");
    LcdBacklightOn();

    IchiAppInitialize(lcd_ready);
    for (;;)
    {
        IchiAppProcess();
        wdt_clear();
        SleepChangetoHaltMode();
    }
}

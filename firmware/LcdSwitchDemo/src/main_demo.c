/*
 * LCD + push-switch bring-up demo for DT-EBML63Q2557 (Solist-AI board).
 *
 * Replaces S_System/main.c in a disposable copy of the private LEXIDE project
 * (AcrylicPanCollector_lowlatency).  Vendor drivers are used as-is and are not
 * vendored in this repository.
 *
 *   SW2..SW5 = P50..P53, active Low (board pull-ups), read every 10 ms.
 *   LED1..LED3 = P54..P56, LED1 = MSB.
 *   LCD = on-board 16x2 I2C character LCD (shares CN3 SCL/SDA).
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
#include "Sw.h"
#include "TimeControl.h"
#include "clock.h"
#include "irq.h"
#include "mcu.h"
#include "smpl_common.h"
#include "smpl_common_led.h"
#include "timer0_1.h"
#include "wdt.h"

#define SW_MASK        (0x0FU)  /* P50..P53 */
#define DEBOUNCE_TICKS (3U)     /* 3 x 10 ms stable */
#define LCD_COLUMNS    (LCD_MOST_CHARACTERS_ON_A_LINE)

static volatile uint8_t pressed_events;   /* bit n = SW(n+2) newly pressed */
static uint8_t stable_level = SW_MASK;
static uint8_t last_raw = SW_MASK;
static uint8_t same_count;

static const char *const messages[4] =
{
    "Hello Solist-AI!",
    "IchiPing ready  ",
    "Stamp-S3A link  ",
    "Interposer test ",
};

/* Called from the vendor 10 ms periodic handler. */
static void poll_switches_10ms(void)
{
    uint8_t raw = (uint8_t)(PORT5->P5DI & SW_MASK);

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
            /* A High->Low transition of a stable level is a new press. */
            pressed_events |= (uint8_t)(stable_level & (uint8_t)~raw);
            stable_level = raw;
        }
    }
}

static void show_leds(uint8_t value)
{
    if ((value & 0x04U) != 0U) { smpl_onLED1(); } else { smpl_offLED1(); }
    if ((value & 0x02U) != 0U) { smpl_onLED2(); } else { smpl_offLED2(); }
    if ((value & 0x01U) != 0U) { smpl_onLED3(); } else { smpl_offLED3(); }
}

/* LcdDraw only writes the characters given, so pad every line to 16 columns. */
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

/* "SW3  count 0012" */
static bool draw_status(uint8_t sw_number, uint16_t count)
{
    char line[LCD_COLUMNS + 1U] = "SW?  count 0000";

    line[2] = (char)('0' + sw_number);
    line[11] = (char)('0' + ((count / 1000U) % 10U));
    line[12] = (char)('0' + ((count / 100U) % 10U));
    line[13] = (char)('0' + ((count / 10U) % 10U));
    line[14] = (char)('0' + (count % 10U));
    return draw_line(LCD_START_OF_SECOND_LINE, line);
}

int32_t main(void)
{
    uint16_t press_count = 0U;
    bool lcd_ok;

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
    show_leds(0x07U);                /* all on until the LCD is up */

    SwInit();                        /* P50..P53 input mode */
    Regulator5VOutputInit();
    Regulator5VOutputOn();           /* backlight supply */
    LcdPeripheralInit();
    LcdInit();
    LcdDisplayOnOff(LCD_DISPLAY_ON, LCD_CURSOR_OFF, LCD_CURSOR_BLINK_OFF);
    LcdClearDisplay();
    lcd_ok = draw_line(LCD_START_OF_FIRST_LINE, "IchiPing Solist");
    lcd_ok = draw_line(LCD_START_OF_SECOND_LINE, "Press SW2-SW5") && lcd_ok;
    LcdBacklightOn();
    show_leds(lcd_ok ? 0x00U : 0x05U);  /* 101 = LCD NACK */

    PeriodicHandler10msSetCallBack(poll_switches_10ms);

    for (;;)
    {
        uint8_t events;

        __disable_irq();
        events = pressed_events;
        pressed_events = 0U;
        __enable_irq();

        if (events != 0U)
        {
            uint8_t index;

            for (index = 0U; index < 4U; index++)
            {
                if ((events & (uint8_t)(1U << index)) != 0U)
                {
                    press_count++;
                    lcd_ok = draw_line(LCD_START_OF_FIRST_LINE, messages[index]);
                    lcd_ok = draw_status((uint8_t)(index + 2U), press_count) && lcd_ok;
                    show_leds(lcd_ok ? (uint8_t)(index + 1U) : 0x05U);
                }
            }
        }
        wdt_clear();
        SleepChangetoHaltMode();
    }
}

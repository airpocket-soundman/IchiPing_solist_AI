/*
 * Data collection helper (tools/build.ps1 -Main ichi_collect_main).
 * Replaces S_System/main.c of the vendor project.
 *
 * The PC drives a session (tools/collect_session.py): it sends 'S' + state (UART1 raw bytes),
 * the Solist moves the PCA9685 servos exactly as ichi_survey_main (UNO Q angles and order,
 * 500 ms hold then release, 500 ms settle) and answers
 *   0x48 MOVED  state u8 | ok u8
 * then the PC records through the Stamp-S3A USB port (firmware/StampMeasure 'M' / 'C' + 'D').
 * 'R' disables the servo outputs.  No feature / inference here.
 */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "Lcd.h"
#include "PeriodicHandler10ms.h"
#include "Regulator5VOutput.h"
#include "SoftwareInterrupt.h"
#include "SystemError.h"
#include "SystemPowerControl.h"
#include "TimeControl.h"
#include "Uart1.h"
#include "clock.h"
#include "ichi_protocol.h"
#include "ichi_servo.h"
#include "ichi_stamp_link.h"
#include "ichi_tft.h"
#include "ichi_ui.h"
#include "mcu.h"
#include "smpl_common.h"
#include "timer0_1.h"
#include "wdt.h"

#define SERVO_HOLD      (50U)          /* x10 ms: hold 500 ms, then release (UNO Q) */
#define SETTLE          (50U)          /* x10 ms after the last move before recording */

#define MSG_MOVED  (0x48U)


static volatile uint32_t tick_10ms;
static volatile bool transmit_busy;
static uint8_t transmit_buffer[ICHI_TX_PAYLOAD_CAPACITY + 48U];
static uint8_t payload[ICHI_TX_PAYLOAD_CAPACITY];
static uint32_t sequence;
static int16_t servo_state = -1;       /* unknown until the first move */
static volatile int16_t move_request = -1;
static volatile bool expect_state;

static void periodic_10ms(void) { tick_10ms++; }

/* PC commands, raw bytes outside the COBS frames: 'S' + state byte (0..31, bit0..4 =
   a b c AB BC, 1 = OPEN) moves the servos; 'R' releases them (0xFF). */
static void receive_byte(uint32_t value, uint16_t error_status)
{
    (void)error_status;
    if (expect_state) { expect_state = false; move_request = (int16_t)(value & 0xFFU); return; }
    if (value == (uint32_t)'S') { expect_state = true; }
    if (value == (uint32_t)'R') { move_request = 0xFF; }
}

static void transmit_complete(uint32_t count, uint16_t error_status)
{
    (void)count;
    (void)error_status;
    transmit_busy = false;
    /* Uart1Write replaces the interrupt-enable register with TX-only bits. */
    Uart1StartReadByte(receive_byte);
}

static void send(uint8_t type, uint16_t size)
{
    size_t encoded;
    while (transmit_busy) { wdt_clear(); }
    encoded = IchiEncodeFrame(type, ++sequence, payload, size, transmit_buffer, sizeof(transmit_buffer));
    if (encoded > 0U)
    {
        transmit_busy = true;
        Uart1Write(transmit_buffer, (uint32_t)encoded, transmit_complete);
    }
    while (transmit_busy) { wdt_clear(); }
}

static void put_u16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put_u32(uint8_t *p, uint32_t v) { put_u16(p, v); put_u16(&p[2], v >> 16); }
static void put_dec(char *d, uint32_t v, uint8_t n) { while (n-- > 0U) { d[n] = (char)('0' + v % 10U); v /= 10U; } }

static void state_label(char *dst, uint8_t state)       /* "h" + C BC B AB A (ichi_ui.c) */
{
    IchiUiStateLabel(dst, state);
}

static void lcd(const char *line1, const char *line2)
{
    char l[17];
    const char *src[2] = {line1, line2};
    uint8_t row, i;
    for (row = 0U; row < 2U; row++)
    {
        for (i = 0U; (i < 16U) && (src[row][i] != '\0'); i++) { l[i] = src[row][i]; }
        for (; i < 16U; i++) { l[i] = ' '; }
        l[16] = '\0';
        (void)LcdDraw(row == 0U ? LCD_START_OF_FIRST_LINE : LCD_START_OF_SECOND_LINE, l);
    }
}



static void wait_ticks(uint32_t n)
{
    uint32_t start = tick_10ms;
    while ((tick_10ms - start) < n)
    {
        wdt_clear();
    }
}

static bool move_one(uint8_t channel, bool open)
{
    if (!IchiServoSetCount(channel, open ? ICHI_SERVO_OPEN_COUNT : ICHI_SERVO_CLOSE_COUNT)) { return false; }
    wait_ticks(SERVO_HOLD);
    return IchiServoRelease(channel);
}

/* Moves only the changed channels: CLOSE BC -> a, then OPEN a -> BC (UNO Q order). */
static bool drive_to(uint8_t target)
{
    int8_t ch;
    bool moved = false;
    IchiServoEnable(true);
    for (ch = 4; ch >= 0; ch--)
    {
        bool want_open = ((target >> ch) & 1U) != 0U;
        bool was_open = (servo_state >= 0) && (((servo_state >> ch) & 1) != 0);
        if (!want_open && ((servo_state < 0) || was_open))
        {
            if (!move_one((uint8_t)ch, false)) { return false; }
            moved = true;
        }
    }
    for (ch = 0; ch < 5; ch++)
    {
        bool want_open = ((target >> ch) & 1U) != 0U;
        bool was_open = (servo_state >= 0) && (((servo_state >> ch) & 1) != 0);
        if (want_open && ((servo_state < 0) || !was_open))
        {
            if (!move_one((uint8_t)ch, true)) { return false; }
            moved = true;
        }
    }
    servo_state = target;
    if (moved) { wait_ticks(SETTLE); }
    return true;
}


int32_t main(void)
{
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
    Uart1PeripheralInit();
    SystemErrorInit();

    Regulator5VOutputInit();
    Regulator5VOutputOn();
    LcdPeripheralInit();
    LcdInit();
    LcdDisplayOnOff(LCD_DISPLAY_ON, LCD_CURSOR_OFF, LCD_CURSOR_BLINK_OFF);
    LcdClearDisplay();
    LcdBacklightOn();

    IchiTftInitialize();
    IchiUiInitialize();
    if (!IchiServoInitialize())
    {
        IchiUiBanner("PCA9685 no ACK", ICHI_RED, ICHI_WHITE);
        lcd("PCA9685 no ACK", "");
        for (;;) { wdt_clear(); }
    }
    IchiUiBanner("Collect: PC control", ICHI_YELLOW, ICHI_BLACK);
    lcd("collect: waiting", "for the PC");
    Uart1StartReadByte(receive_byte);

    for (;;)
    {
        int16_t request = move_request;
        if (request >= 0)
        {
            char line[] = "h00000";
            bool ok;
            move_request = -1;
            if (request == 0xFF)
            {
                IchiServoEnable(false);
                servo_state = -1;
                ok = true;
            }
            else
            {
                uint8_t state = (uint8_t)(request & 0x1F);
                state_label(line, state);
                IchiUiBanner("Moving...", ICHI_YELLOW, ICHI_BLACK);
                lcd("collect: move", line);
                ok = drive_to(state);
                IchiUiShowState(state, ICHI_UI_NONE);
                IchiUiBanner(ok ? "Recording (PC)" : "Servo error", ok ? ICHI_CYAN : ICHI_RED, ICHI_BLACK);
            }
            payload[0] = (uint8_t)request;
            payload[1] = ok ? 1U : 0U;
            send(MSG_MOVED, 2U);
        }
        wait_ticks(1U);
    }
}

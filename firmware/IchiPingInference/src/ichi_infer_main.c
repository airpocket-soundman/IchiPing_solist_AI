/*
 * IchiPing inference firmware (tools/build.ps1 -Main ichi_infer_main).
 * Replaces S_System/main.c of the vendor project.
 *
 * Behaviour follows the original IchiPing 10_inference (and its UNO Q port):
 *   power on  -> all servos CLOSE -> baseline (3 all-closed frames, "Calibrating...")
 *   switches  -> the servos follow them (manual door / window operation); a change of the
 *                actual state clears the previous result ("-----", banner cleared)
 *   EXEC      -> one inference: PRBS play + record on the Stamp (firmware/StampMeasure),
 *                N333 feature + int8 CNN front-end on the CPU, ELM head on the AxlCORE;
 *                inf / act digit rows and the Complete / Conditional Success / Failure banner
 * Solist-AI addition, on-site calibration (docs/HANDOFF_SOLIST_CNN_FRONTEND_20260925.md):
 *   EXEC held 2 s -> the servos visit all 32 states (Gray code); per state one 6 s PRBS
 *                gives five 2 s windows (1 s hop) and each window is one OS-ELM update of the
 *                ELM beta on the AxlCORE, starting from the factory beta and
 *                P0 = w (G_f + lambda I)^-1 (generated/ichi_calib_prior.h).
 *                EXEC during calibration aborts and restores the factory beta.
 *   The calibrated beta lasts until power off.
 * UART1 (tools/infer_monitor.py), frames as in ichi_protocol.h:
 *   0x50 BASE     index u8 | exponent u8 | transfer ms u32
 *   0x51 RESULT   actual u8 | pred u8 | calibrated u8 | exponent u8 | transfer ms u32 |
 *                 outputs float32 x 32
 *   0x52 CAL_STEP state u8 | window u8 | pred before update u8 | 0 | transfer ms u32
 *   0x53 CAL_DONE correct-before u16 | samples u16 | aborted u8 | seconds u16
 *   0x54 ERROR    stage u8 | state u8  (1 Stamp, 3 measure, 4 PCM, 5 inference, 6 servo, 7 train)
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
#include "ichi_calib_prior.h"
#include "ichi_feature.h"
#include "ichi_inference.h"
#include "ichi_protocol.h"
#include "ichi_servo.h"
#include "ichi_stamp_link.h"
#include "ichi_tft.h"
#include "ichi_ui.h"
#include "mcu.h"
#include "smpl_common.h"
#include "timer0_1.h"
#include "wdt.h"

#define BASELINE_FRAMES (3U)
#define FRAME_SAMPLES   (32000U)
#define SERVO_HOLD      (50U)          /* x10 ms: hold 500 ms, then release (UNO Q) */
#define SETTLE          (50U)          /* x10 ms after the last move before recording */
#define LONG_PRESS      (200U)         /* x10 ms: EXEC held this long starts calibration */

#define MSG_BASE     (0x50U)
#define MSG_RESULT   (0x51U)
#define MSG_CAL_STEP (0x52U)
#define MSG_CAL_DONE (0x53U)
#define MSG_ERROR    (0x54U)

static volatile uint32_t tick_10ms;
static volatile bool transmit_busy;
static uint8_t transmit_buffer[ICHI_TX_PAYLOAD_CAPACITY + 48U];
static uint8_t payload[ICHI_TX_PAYLOAD_CAPACITY];
static int8_t feature[ICHI_FEAT_INPUTS];
static uint32_t sequence;
static int16_t servo_state = -1;       /* unknown until the first move */
static uint8_t switch_state;
static bool exec_down;
static bool abort_request;             /* new EXEC press during a long operation */
static bool calibrated;

static void periodic_10ms(void) { tick_10ms++; }

static void transmit_complete(uint32_t count, uint16_t error_status)
{
    (void)count;
    (void)error_status;
    transmit_busy = false;
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

/* Reads the switches and EXEC from the Stamp; a new EXEC press sets abort_request. */
static bool poll_switches(void)
{
    uint8_t sw;
    bool pressed;
    if (!IchiStampGetSwitches(&sw, &pressed)) { return false; }
    switch_state = sw;
    if (pressed && !exec_down) { abort_request = true; }
    exec_down = pressed;
    return true;
}

static void wait_ticks(uint32_t n)
{
    uint32_t start = tick_10ms, last = tick_10ms;
    while ((tick_10ms - start) < n)
    {
        if ((tick_10ms - last) >= 5U) { last = tick_10ms; (void)poll_switches(); }
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

static void report_error(uint8_t stage, uint8_t state)
{
    char text[] = "Error stage 0";
    char line[] = "h00000";
    payload[0] = stage;
    payload[1] = state;
    send(MSG_ERROR, 2U);
    text[12] = (char)('0' + stage);
    state_label(line, state);
    IchiUiBanner(text, ICHI_RED, ICHI_WHITE);
    lcd(text, line);
}

/* Reads window `clip` of the last measurement into the feature buffer (0 = ok, else stage). */
static uint8_t read_window(uint8_t clip, uint32_t *transfer_ms)
{
    uint32_t t0 = tick_10ms;
    IchiStampSelectClip(clip);
    if (!IchiFeatureFrame(IchiStampReadPcm, FRAME_SAMPLES)) { return 4U; }
    *transfer_ms = (tick_10ms - t0) * 10UL;
    return 0U;
}

static bool baseline(void)
{
    uint8_t i;
    IchiUiBanner("Calibrating...", ICHI_CYAN, ICHI_BLACK);
    lcd("baseline", "all closed");
    IchiFeatureBaselineReset();
    for (i = 0U; i < BASELINE_FRAMES; i++)
    {
        char info[] = "baseline 0/0";
        uint32_t ms = 0U;
        uint8_t stage;
        info[9] = (char)('1' + i);
        info[11] = (char)('0' + BASELINE_FRAMES);
        IchiUiInfoLine(info, ICHI_CYAN);
        if (!IchiStampMeasure(ICHI_MEASURE_FRAME)) { report_error(3U, 0U); return false; }
        stage = read_window(0U, &ms);
        if (stage != 0U) { report_error(stage, 0U); return false; }
        IchiFeatureBaselineAdd(BASELINE_FRAMES);
        payload[0] = i;
        payload[1] = IchiFeatureLastExponent();
        put_u32(&payload[2], ms);
        send(MSG_BASE, 6U);
    }
    IchiUiInfoLine(calibrated ? "calibrated" : "factory model", ICHI_WHITE);
    IchiUiBanner("Baseline ready", ICHI_GREEN, ICHI_WHITE);
    lcd("baseline ready", "EXEC: infer");
    return true;
}

static void infer(void)
{
    float output[ICHI_INFERENCE_OUTPUT_COUNT];
    uint32_t ms = 0U;
    uint8_t pred = 0U, stage, k;
    uint8_t actual = (uint8_t)servo_state;
    char line1[] = "act h00000";
    char line2[] = "inf h00000 NG";

    IchiUiShowState(actual, ICHI_UI_NONE);
    IchiUiBanner("Listening...", ICHI_CYAN, ICHI_BLACK);
    lcd("Listening...", "");
    if (!IchiStampMeasure(ICHI_MEASURE_FRAME)) { report_error(3U, actual); return; }
    stage = read_window(0U, &ms);
    if (stage != 0U) { report_error(stage, actual); return; }
    IchiFeatureInput(feature);
    if (!IchiInferenceRun((const uint8_t *)feature, output, &pred)) { report_error(5U, actual); return; }

    IchiUiShowState(actual, pred);
    state_label(&line1[4], actual);
    state_label(&line2[4], pred);
    if (pred == actual) { line2[11] = 'O'; line2[12] = 'K'; }
    lcd(line1, line2);
    payload[0] = actual;
    payload[1] = pred;
    payload[2] = calibrated ? 1U : 0U;
    payload[3] = IchiFeatureLastExponent();
    put_u32(&payload[4], ms);
    for (k = 0U; k < ICHI_INFERENCE_OUTPUT_COUNT; k++) { memcpy(&payload[8U + 4U * k], &output[k], 4U); }
    send(MSG_RESULT, (uint16_t)(8U + 4U * ICHI_INFERENCE_OUTPUT_COUNT));
}

static void calibrate(void)
{
    uint32_t start = tick_10ms;
    uint16_t correct = 0U, samples = 0U;
    uint8_t i, w, stage = 0U;

    abort_request = false;
    IchiUiBanner("On-site calibration", ICHI_MAGENTA, ICHI_WHITE);
    lcd("calibration", "EXEC: abort");
    (void)IchiInferenceCalibrationBegin();
    calibrated = false;
    for (i = 0U; (i < 32U) && (stage == 0U) && !abort_request; i++)
    {
        uint8_t state = (uint8_t)(i ^ (i >> 1));                 /* Gray code: one servo per step */
        char info[] = "cal 00/32 h00000";
        put_dec(&info[4], (uint32_t)i + 1U, 2U);
        state_label(&info[10], state);
        IchiUiInfoLine(info, ICHI_MAGENTA);
        lcd(info, "EXEC: abort");
        if (!drive_to(state)) { stage = 6U; report_error(6U, state); break; }
        IchiUiShowState(state, ICHI_UI_NONE);
        if (abort_request) { break; }
        if (!IchiStampMeasure(ICHI_MEASURE_CALIBRATION)) { stage = 3U; report_error(3U, state); break; }
        for (w = 0U; (w < ICHI_CAL_PER_STATE) && !abort_request; w++)
        {
            float output[ICHI_INFERENCE_OUTPUT_COUNT];
            uint32_t ms = 0U;
            uint8_t pred = 0U;
            stage = read_window(w, &ms);
            if (stage != 0U) { report_error(stage, state); break; }
            IchiFeatureInput(feature);
            (void)IchiInferenceRun((const uint8_t *)feature, output, &pred);   /* before the update */
            if (!IchiInferenceTrain((const uint8_t *)feature, state)) { stage = 7U; report_error(7U, state); break; }
            samples++;
            if (pred == state) { correct++; }
            payload[0] = state;
            payload[1] = w;
            payload[2] = pred;
            payload[3] = 0U;
            put_u32(&payload[4], ms);
            send(MSG_CAL_STEP, 8U);
            (void)poll_switches();
        }
    }
    if (abort_request || (stage != 0U))
    {
        IchiInferenceUseFactory();
        IchiUiBanner(abort_request ? "Calibration aborted" : "Calibration failed", ICHI_RED, ICHI_WHITE);
        IchiUiInfoLine("factory model", ICHI_WHITE);
    }
    else
    {
        calibrated = true;
        IchiUiBanner("Calibrated", ICHI_GREEN, ICHI_WHITE);
        {
            char info[] = "calibrated 000/000";
            put_dec(&info[11], correct, 3U);
            put_dec(&info[15], samples, 3U);
            IchiUiInfoLine(info, ICHI_WHITE);
        }
    }
    put_u16(&payload[0], correct);
    put_u16(&payload[2], samples);
    payload[4] = abort_request ? 1U : 0U;
    put_u16(&payload[5], (tick_10ms - start) / 100UL);
    send(MSG_CAL_DONE, 7U);
    abort_request = false;
    lcd(calibrated ? "calibrated" : "factory model", "EXEC: infer");
}

int32_t main(void)
{
    uint32_t press_start = 0U;
    bool was_down = false, long_done = false;

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
    SysTick->LOAD = 0x00FFFFFFUL;
    SysTick->VAL = 0UL;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_ENABLE_Msk;
    IchiInferenceInitialize();

    IchiTftInitialize();
    IchiUiInitialize();
    if (!IchiServoInitialize())
    {
        IchiUiBanner("PCA9685 no ACK", ICHI_RED, ICHI_WHITE);
        lcd("PCA9685 no ACK", "");
        for (;;) { wdt_clear(); }
    }

    /* Power on: all closed, then the baseline (original IchiPing CALIBRATING step). */
    IchiUiBanner("Closing all", ICHI_YELLOW, ICHI_BLACK);
    lcd("closing all", "");
    while (!poll_switches())
    {
        IchiUiBanner("Stamp not ready", ICHI_YELLOW, ICHI_BLACK);
        wait_ticks(50U);
    }
    if (!drive_to(0U)) { report_error(6U, 0U); }
    IchiUiShowState(0U, ICHI_UI_NONE);
    while (!baseline())
    {
        wait_ticks(200U);
    }
    exec_down = true;                  /* ignore an EXEC held since power on */
    was_down = true;
    long_done = true;

    for (;;)
    {
        static uint32_t last_poll = 0U;
        if (tick_10ms != last_poll) { last_poll = tick_10ms; (void)poll_switches(); }   /* every 10 ms */

        /* The servos follow the switches; a changed state clears the previous result. */
        if ((servo_state >= 0) && (switch_state != (uint8_t)servo_state) && !exec_down)
        {
            IchiUiBanner("", ICHI_BLACK, ICHI_BLACK);
            if (!drive_to(switch_state)) { report_error(6U, switch_state); }
            IchiUiShowState((uint8_t)servo_state, ICHI_UI_NONE);
            {
                char line1[] = "act h00000";
                state_label(&line1[4], (uint8_t)servo_state);
                lcd(line1, "EXEC: infer");
            }
        }

        /* EXEC: short press = inference, held LONG_PRESS = on-site calibration. */
        if (exec_down && !was_down) { press_start = tick_10ms; long_done = false; }
        if (exec_down && !long_done && ((tick_10ms - press_start) >= LONG_PRESS))
        {
            long_done = true;
            calibrate();
            (void)poll_switches();
            if (switch_state != (uint8_t)servo_state)          /* back to the switch state */
            {
                if (!drive_to(switch_state)) { report_error(6U, switch_state); }
            }
            IchiUiShowState((uint8_t)servo_state, ICHI_UI_NONE);
        }
        if (!exec_down && was_down && !long_done)
        {
            abort_request = false;
            infer();
        }
        was_down = exec_down;
        abort_request = false;
        wdt_clear();
    }
}

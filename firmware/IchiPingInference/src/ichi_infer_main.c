/*
 * IchiPing inference firmware (tools/build.ps1 -Main ichi_infer_main).
 * Replaces S_System/main.c of the vendor project.
 *
 * Behaviour follows the original IchiPing 10_inference (and its UNO Q port):
 *   power on  -> all servos CLOSE -> baseline (3 all-closed frames, "Calibrating...")
 *   switches  -> the act digits follow them at once (within ~0.1 s) and the previous result
 *                is cleared; the servos catch up in the background (manual door / window
 *                operation), one channel at a time without blocking the loop
 *   EXEC      -> waits for the servos to finish and settle, then one inference: PRBS play +
 *                record on the Stamp (firmware/StampMeasure), N333 feature + int8 CNN front-end
 *                on the CPU, ELM head on the AxlCORE; inf / act digit rows and the
 *                Complete / Conditional Success / Failure banner
 * Solist-AI addition, on-site calibration (docs/DEVELOPMENT.md), only
 * needed after moving the device to a new place (the pre-trained model covers this rig):
 *   EXEC held 2 s -> show add-ODL hint; release -> confirmation. All switches OPEN
 *                plus a new EXEC press starts ODL; any other switch state cancels.
 *                A 10 s hold requests reset instead, with the same confirmation.
 *                Confirmed reset clears saved ODL and restores the factory model.
 *                ODL visits all 32 states (Gray code); per state one 6 s PRBS
 *                gives five 2 s windows (1 s hop) and each window is one OS-ELM update of the
 *                ELM beta (float32 on the CPU, P and beta in the FRAM, see ichi_inference.c),
 *                continuing the saved beta/P, or starting from pre-trained beta
 *                and P0 if no valid saved state exists (generated/ichi_calib_prior.h).
 *                EXEC during calibration aborts and restores the last committed ODL,
 *                or the factory model when no saved ODL exists.
 *   Successful calibration is saved to FRAM and restored on boot if the
 *   model identity and snapshot checksum match.
 * UART1 (tools/infer_monitor.py): a raw 'I' byte from the PC starts one inference (like EXEC);
 * frames as in ichi_protocol.h:
 *   0x50 BASE     index u8 | exponent u8 | transfer ms u32
 *   0x51 RESULT   actual u8 | pred u8 | calibrated u8 | exponent u8 | transfer ms u32 |
 *                 outputs float32 x 32
 *   0x52 CAL_STEP state u8 | window u8 | pred after update u8 | 1 | transfer ms u32
 *   0x53 CAL_DONE correct-after u16 | samples u16 | aborted u8 | seconds u16 | 1
 *   0x54 ERROR    stage u8 | state u8  (1 Stamp, 3 measure, 4 PCM, 5 inference, 6 servo, 7 train)
 */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "Lcd.h"
#include "HighSpeedCom.h"
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
#include "ichi_exec_control.h"
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
#define ODL_SETTLE      (10U)          /* x10 ms: pulse already held 500 ms; only 100 ms acoustic settling */
#define ODL_WINDOWS     (5U)           /* retain all five labelled windows for every state */

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

/* The vendor SAD interrupt references its legacy IEPE-to-SPI streaming API.
   This application never enables IEPE streaming or initializes that SPI slave:
   audio comes from Stamp over I2C, and PC results use UART. Avoid retaining the
   unused vendor streaming library and its 1 KB FIFO; use that RAM for PCM reuse. */
HIGH_SPEED_COM_RESULT HighSpeedComInputRtData(uint16_t data)
{
    (void)data;
    return HIGH_SPEED_COM_RESULT_BUSY;
}

static void periodic_10ms(void) { tick_10ms++; }

static volatile bool pc_infer_request;
static volatile bool pc_calibration_request;

/* PC command, a raw byte outside the COBS frames: 'I' = one inference (same as an EXEC press). */
static void receive_byte(uint32_t value, uint16_t error_status)
{
    (void)error_status;
    if (value == (uint32_t)'I') { pc_infer_request = true; }
    if (value == (uint32_t)'C') { pc_calibration_request = true; }
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

/* ---- Servos following the switches without blocking --------------------------------
   The screen shows a switch change at once (<= ~0.1 s: 10 ms STATUS poll + 20 ms debounce
   on the Stamp + ~20 ms of TFT digits); the servos catch up in the background, one channel
   at a time in the UNO Q order (CLOSE BC -> a, then OPEN a -> BC), each held SERVO_HOLD and
   released.  An inference waits until the servos are idle and SETTLE has passed. */
static int8_t moving_channel = -1;
static uint32_t move_release_tick;
static uint32_t settle_until;
static uint8_t servo_target;
static int16_t shown_actual = -1;

static int8_t next_channel(uint8_t target)
{
    int8_t ch;
    for (ch = 4; ch >= 0; ch--)                          /* closes first, BC -> a */
    {
        if ((((target >> ch) & 1U) == 0U) && (((servo_state >> ch) & 1) != 0)) { return ch; }
    }
    for (ch = 0; ch < 5; ch++)                           /* then opens, a -> BC */
    {
        if ((((target >> ch) & 1U) != 0U) && (((servo_state >> ch) & 1) == 0)) { return ch; }
    }
    return -1;
}

/* One step of the background servo motion; returns false on a PCA9685 error. */
static bool servo_step(void)
{
    if (moving_channel >= 0)
    {
        if ((int32_t)(tick_10ms - move_release_tick) < 0) { return true; }
        if (!IchiServoRelease((uint8_t)moving_channel)) { moving_channel = -1; return false; }
        servo_state = (int16_t)(servo_state ^ (int16_t)(1 << moving_channel));
        moving_channel = -1;
        if ((uint8_t)servo_state == servo_target) { settle_until = tick_10ms + SETTLE; }
    }
    if ((servo_state >= 0) && ((uint8_t)servo_state != servo_target))
    {
        int8_t ch = next_channel(servo_target);
        bool open = ((servo_target >> ch) & 1U) != 0U;
        IchiServoEnable(true);
        if (!IchiServoSetCount((uint8_t)ch, open ? ICHI_SERVO_OPEN_COUNT : ICHI_SERVO_CLOSE_COUNT)) { return false; }
        moving_channel = ch;
        move_release_tick = tick_10ms + SERVO_HOLD;
    }
    return true;
}

static bool servos_idle(void)
{
    return (moving_channel < 0) && (servo_state >= 0) && ((uint8_t)servo_state == servo_target) &&
           ((int32_t)(tick_10ms - settle_until) >= 0);
}

/* Blocks until the background motion has finished and settled (before measuring). */
static bool servo_finish(void)
{
    while (!servos_idle())
    {
        if (!servo_step()) { return false; }
        wdt_clear();
    }
    return true;
}

static void show_switch_state(uint8_t state)
{
    char line1[] = "act h00000";
    if ((int16_t)state == shown_actual) { return; }
    IchiUiShowState(state, ICHI_UI_NONE);                /* digits first: that is what the user watches */
    IchiUiBanner("", ICHI_BLACK, ICHI_BLACK);            /* the previous result no longer applies */
    state_label(&line1[4], state);
    lcd(line1, "EXEC: infer");
    shown_actual = (int16_t)state;
}

static bool move_one(uint8_t channel, bool open)
{
    if (!IchiServoSetCount(channel, open ? ICHI_SERVO_OPEN_COUNT : ICHI_SERVO_CLOSE_COUNT)) { return false; }
    wait_ticks(SERVO_HOLD);
    return IchiServoRelease(channel);
}

/* Moves only the changed channels: CLOSE BC -> a, then OPEN a -> BC (UNO Q order). */
static bool drive_to(uint8_t target, uint32_t settle_ticks)
{
    int8_t ch;
    bool moved = false;
    servo_target = target;
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
    if (moved) { wait_ticks(settle_ticks); }
    settle_until = tick_10ms;
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

/* Transfer includes the streaming Welch FFT; final normalization and model run are
   presented as inference. Only model completion fills the gauge completely. */
#define GAUGE_TRANSFER_TEXT  "Transferring..."
#define GAUGE_INFERENCE_TEXT "Inferring..."
#define GAUGE_BACK  ICHI_RGB565(0, 0, 72)
#define GAUGE_FILL  ICHI_RGB565(0, 150, 255)
#define GAUGE_CHUNK ICHI_STAMP_READ_MAX   /* one I2C READ per gauge step (~270 steps) */
/* The feature extractor retains overlapping PCM in RAM and reads each sample once. */
#define GAUGE_SEGMENTS (((FRAME_SAMPLES - ICHI_FEAT_NFFT) / ICHI_FEAT_HOP) + 1UL)
#define GAUGE_TOTAL (ICHI_FEAT_NFFT + (GAUGE_SEGMENTS - 1UL) * ICHI_FEAT_HOP)
static bool gauge_on;
static uint32_t gauge_read;
static const char *gauge_stage_text = GAUGE_TRANSFER_TEXT;

static void gauge(uint16_t permille)
{
    /* Software-SPI redraws are expensive on the M0+.  Quantize only the
       presentation to 5% steps; PCM reads and inference are unchanged.
       Reset on zero so every inference starts a fresh progress display. */
    static uint16_t last_drawn;
    static const char *last_stage;
    uint16_t step = (uint16_t)(permille / 50U);
    if ((permille != 0U) && (permille != 1000U) && (step == last_drawn) && (last_stage == gauge_stage_text)) { return; }
    last_drawn = step;
    last_stage = gauge_stage_text;
    IchiUiProgress(gauge_stage_text, permille, GAUGE_FILL, GAUGE_BACK, ICHI_WHITE);
}

/* IchiStampReadPcm in GAUGE_CHUNK pieces so the gauge moves smoothly. */
static bool read_pcm(uint16_t offset, int16_t *dst, uint16_t count)
{
    while (count > 0U)
    {
        uint16_t n = (count > GAUGE_CHUNK) ? GAUGE_CHUNK : count;
        if (!IchiStampReadPcm(offset, dst, n)) { return false; }
        offset = (uint16_t)(offset + n);
        dst += n;
        count = (uint16_t)(count - n);
        gauge_read += n;
        if (gauge_on)
        {
            uint32_t progress = (gauge_read * 900UL) / GAUGE_TOTAL;
            /* Only successful IchiInferenceRun() may advance the gauge to 100%. */
            gauge((uint16_t)((progress > 900UL) ? 900UL : progress));
        }
    }
    return true;
}

/* Reads window `clip` of the last measurement into the feature buffer (0 = ok, else stage). */
static uint8_t read_window(uint8_t clip, uint32_t *transfer_ms)
{
    uint32_t t0 = tick_10ms;
    IchiStampSelectClip(clip);
    if (!IchiFeatureFrame(read_pcm, FRAME_SAMPLES)) { return 4U; }
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
    IchiUiInfoLine(calibrated ? "calibrated" : "pretrained", ICHI_WHITE);
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
    gauge_stage_text = GAUGE_TRANSFER_TEXT;
    gauge(0U);                                           /* the sound has stopped */
    lcd("Transferring...", "");
    gauge_on = true;
    gauge_read = 0U;
    stage = read_window(0U, &ms);
    gauge_on = false;
    if (stage != 0U) { report_error(stage, actual); return; }
    gauge_stage_text = GAUGE_INFERENCE_TEXT;
    gauge(900U);
    lcd("Inferring...", "");
    IchiFeatureInput(feature);
    gauge(950U);
    if (!IchiInferenceRun((const uint8_t *)feature, output, &pred)) { report_error(5U, actual); return; }
    gauge(1000U);

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
    bool continued = false;

    abort_request = false;
    if (!IchiInferenceCalibrationContinue(&continued)) { stage = 7U; report_error(7U, 0U); }
    IchiUiBanner(continued ? "Additional ODL" : "New ODL", ICHI_MAGENTA, ICHI_WHITE);
    lcd(continued ? "additional ODL" : "new ODL", "EXEC: abort");
    calibrated = false;
    for (i = 0U; (i < 32U) && (stage == 0U) && !abort_request; i++)
    {
        uint8_t state = (uint8_t)(i ^ (i >> 1));                 /* Gray code: one servo per step */
        char info[] = "cal 00/32 h00000";
        put_dec(&info[4], (uint32_t)i + 1U, 2U);
        state_label(&info[10], state);
        IchiUiInfoLine(info, ICHI_MAGENTA);
        lcd(info, "EXEC: abort");
        if (!drive_to(state, ODL_SETTLE)) { stage = 6U; report_error(6U, state); break; }
        IchiUiShowState(state, ICHI_UI_NONE);
        if (abort_request) { break; }
        if (!IchiStampMeasure(ICHI_MEASURE_CALIBRATION)) { stage = 3U; report_error(3U, state); break; }
        for (w = 0U; (w < ODL_WINDOWS) && !abort_request; w++)
        {
            float output[ICHI_INFERENCE_OUTPUT_COUNT];
            uint32_t ms = 0U;
            uint8_t pred = 0U;
            stage = read_window(w, &ms);
            if (stage != 0U) { report_error(stage, state); break; }
            IchiFeatureInput(feature);
            if (!IchiInferenceTrain((const uint8_t *)feature, state)) { stage = 7U; report_error(7U, state); break; }
            if (!IchiInferenceRun((const uint8_t *)feature, output, &pred)) { stage = 5U; report_error(5U, state); break; }
            samples++;
            if (pred == state) { correct++; }
            payload[0] = state;
            payload[1] = w;
            payload[2] = pred;
            payload[3] = 1U; /* prediction is after this sample's update */
            put_u32(&payload[4], ms);
            send(MSG_CAL_STEP, 8U);
            (void)poll_switches();
        }
    }
    if (abort_request || (stage != 0U))
    {
        calibrated = IchiInferenceCalibrationRestore();
        if (!calibrated) { IchiInferenceUseFactory(); }
        IchiUiBanner(abort_request ? "Calibration aborted" : "Calibration failed", ICHI_RED, ICHI_WHITE);
        IchiUiInfoLine(calibrated ? "Previous ODL restored" : "pretrained", ICHI_WHITE);
    }
    else
    {
        calibrated = IchiInferenceCalibrationSave();
        if (!calibrated) { stage = 7U; report_error(7U, 0U); IchiInferenceUseFactory(); }
        else { IchiUiBanner("Calibrated", ICHI_GREEN, ICHI_WHITE); }
        {
            char info[] = "post-update 000/000";
            put_dec(&info[12], correct, 3U);
            put_dec(&info[16], samples, 3U);
            IchiUiInfoLine(info, ICHI_WHITE);
        }
    }
    put_u16(&payload[0], correct);
    put_u16(&payload[2], samples);
    payload[4] = (abort_request || (stage != 0U)) ? 1U : 0U;
    put_u16(&payload[5], (tick_10ms - start) / 100UL);
    payload[7] = 1U; /* correct-after, rather than legacy correct-before */
    send(MSG_CAL_DONE, 8U);
    abort_request = false;
    lcd(calibrated ? "calibrated" : "pretrained", "EXEC: infer");
}

int32_t main(void)
{
    IchiExecControl exec_control;

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
    calibrated = IchiInferenceCalibrationRestore();

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
    if (!drive_to(0U, SETTLE)) { report_error(6U, 0U); }
    IchiUiShowState(0U, ICHI_UI_NONE);
    while (!baseline())
    {
        wait_ticks(200U);
    }
    Uart1StartReadByte(receive_byte);
    exec_down = true;                  /* ignore an EXEC held since power on */
    IchiExecControlInit(&exec_control, true);

    for (;;)
    {
        static uint32_t last_poll = 0U;
        IchiExecAction action;
        if (tick_10ms != last_poll) { last_poll = tick_10ms; (void)poll_switches(); }   /* every 10 ms */

        /* The screen follows the switches at once; the servos catch up in the background. */
        if ((servo_state >= 0) && !exec_down && (exec_control.pending == 0U))
        {
            show_switch_state(switch_state);
            servo_target = switch_state;
        }
        if (!servo_step()) { report_error(6U, servo_target); }

        action = IchiExecControlUpdate(&exec_control, tick_10ms, exec_down, switch_state == 31U);
        if (pc_calibration_request)
        {
            pc_calibration_request = false;
            IchiExecRequestAdd(&exec_control, exec_down);
            action = ICHI_EXEC_CONFIRM_ADD;
        }
        if (action == ICHI_EXEC_SHOW_ADD)
        {
            IchiUiBanner("Release: add ODL", ICHI_MAGENTA, ICHI_WHITE);
            IchiUiInfoLine("Hold 10s: reset ODL", ICHI_WHITE);
            lcd("Release: add ODL", "Hold 10s: reset");
        }
        if ((action == ICHI_EXEC_CONFIRM_ADD) || (action == ICHI_EXEC_CONFIRM_RESET))
        {
            IchiUiBanner((action == ICHI_EXEC_CONFIRM_ADD) ? "Add ODL?" : "Reset ODL?", ICHI_MAGENTA, ICHI_WHITE);
            IchiUiInfoLine("All OPEN + EXEC: yes", ICHI_WHITE);
            lcd("All OPEN: yes", "other: no + EXEC");
        }
        if (action == ICHI_EXEC_CANCEL)
        {
            IchiUiBanner("Canceled", ICHI_YELLOW, ICHI_BLACK);
            IchiUiInfoLine(calibrated ? "calibrated" : "pretrained", ICHI_WHITE);
            lcd("Canceled", "EXEC: infer");
        }
        if (action == ICHI_EXEC_RESET_ODL)
        {
            if (IchiInferenceCalibrationClear()) {
                calibrated = false;
                IchiUiBanner("ODL reset", ICHI_GREEN, ICHI_WHITE);
                IchiUiInfoLine("Pretrained model", ICHI_WHITE);
                lcd("ODL reset", "Hold 2s: ODL");
            } else { report_error(7U, (uint8_t)servo_state); }
        }
        if (action == ICHI_EXEC_START_ADD)
        {
            if (!servo_finish()) { report_error(6U, servo_target); }
            else { calibrate(); }
            (void)poll_switches();
            IchiExecControlInit(&exec_control, exec_down);
            if (switch_state != (uint8_t)servo_state)          /* back to the switch state */
            {
                if (!drive_to(switch_state, SETTLE)) { report_error(6U, switch_state); }
            }
            IchiUiShowState((uint8_t)servo_state, ICHI_UI_NONE);
            shown_actual = servo_state;
        }
        if ((action == ICHI_EXEC_INFER) || (pc_infer_request && exec_control.pending == 0U))
        {
            pc_infer_request = false;
            abort_request = false;
            if (!servo_finish()) { report_error(6U, servo_target); }
            else { infer(); }
        }
        abort_request = false;
        wdt_clear();
    }
}

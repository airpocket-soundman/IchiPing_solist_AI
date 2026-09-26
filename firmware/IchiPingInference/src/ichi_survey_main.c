/*
 * 32-state on-board survey (tools/build.ps1 -Main ichi_survey_main).
 * Replaces S_System/main.c of the vendor project.
 *
 * Everything the product does, driven state by state:
 *   servos (PCA9685, UNO Q angles) set the door/window state ->
 *   Stamp-S3A plays the PRBS and records the INMP441 (firmware/StampMeasure) ->
 *   the Solist reads the aligned 2 s frame over I2C, computes the N333 feature
 *   (ichi_feature.c), runs the int8 CNN front-end + ELM (ichi_inference.c).
 * Sequence: EXEC or the PC starts (a raw 'G' byte on UART1; tools/survey_monitor.py sends it, 'X'
 * aborts like EXEC); all closed -> SURVEY_BASELINE frames -> (SURVEY_CALIBRATE) on-site
 * calibration: 32 states x one 6 s PRBS x 5 windows, one OS-ELM update each on the AxlCORE
 * (ichi_infer_main.c does the same on a long EXEC press) -> SURVEY_ROUNDS rounds
 * of the 32 states in Gray-code order (UNO Q collector order: round r starts at
 * (r*11)%32, odd rounds reversed).  EXEC during the survey aborts (OE High).
 * The TFT uses the original IchiPing inference screen (ichi_ui.c: inf / act digit rows
 * and the Complete / Conditional Success / Failure banner) plus a progress line.
 * Results also go to the character LCD and UART1 (tools/survey_monitor.py):
 *   0x40 START  rounds u8 | baseline u8 | states u8 | calibrate u8
 *   0x41 BASE   index u8 | exponent u8 | transfer ms u32
 *   0x42 RESULT state u8 | round u8 | pred u8 | pred_factory u8 | transfer ms u32 |
 *               feature ms u16 | infer ms u16 | outputs float32 x 32
 *               (pred = AxlCORE with the current beta; pred_factory = factory head on the CPU)
 *   0x43 DONE   correct u16 | total u16 | aborted u8 | seconds u16 | correct_factory u16
 *   0x45 CAL_STEP state u8 | window u8 | pred before update u8 | 0 | transfer ms u32
 *   0x46 CAL_DONE correct-before u16 | samples u16 | ok u8 | seconds u16
 *   0x44 ERROR  stage u8 | state u8   (1 Stamp, 3 measure, 4 PCM, 5 inference, 6 servo)
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

#define SURVEY_BASELINE (3U)
#define SURVEY_ROUNDS   (1U)
#define SURVEY_CALIBRATE (0U)          /* 1: on-site calibration before the survey */
#define SURVEY_STATES   (32U)
#define FRAME_SAMPLES   (32000U)
#define SERVO_HOLD      (50U)          /* x10 ms: hold 500 ms, then release (UNO Q) */
#define SETTLE          (50U)          /* x10 ms after the last move before recording */

#define MSG_START  (0x40U)
#define MSG_BASE   (0x41U)
#define MSG_RESULT (0x42U)
#define MSG_DONE   (0x43U)
#define MSG_ERROR  (0x44U)
#define MSG_CAL_STEP (0x45U)
#define MSG_CAL_DONE (0x46U)


static volatile uint32_t tick_10ms;
static volatile bool transmit_busy;
static uint8_t transmit_buffer[ICHI_TX_PAYLOAD_CAPACITY + 48U];
static uint8_t payload[ICHI_TX_PAYLOAD_CAPACITY];
static int8_t feature[ICHI_FEAT_INPUTS];
static uint32_t sequence;
static int16_t servo_state = -1;       /* unknown until the first move */
static bool exec_down;
static volatile bool abort_request;
static volatile bool start_request;

static void periodic_10ms(void) { tick_10ms++; }

/* PC commands: single raw bytes outside the COBS frames ('G' start, 'X' abort). */
static void receive_byte(uint32_t value, uint16_t error_status)
{
    (void)error_status;
    if (value == (uint32_t)'G') { start_request = true; }
    if (value == (uint32_t)'X') { abort_request = true; }
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


/* Polls EXEC (Stamp STATUS); a new press requests an abort. */
static void check_exec(void)
{
    uint8_t sw;
    bool pressed;
    if (IchiStampGetSwitches(&sw, &pressed))
    {
        if (pressed && !exec_down) { abort_request = true; }
        exec_down = pressed;
    }
}

static void wait_ticks(uint32_t n)
{
    uint32_t start = tick_10ms, last = tick_10ms;
    while ((tick_10ms - start) < n)
    {
        if ((tick_10ms - last) >= 5U) { last = tick_10ms; check_exec(); }
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
    lcd("survey ERROR", line);
}

/* One live frame into the feature buffer; returns the stage that failed (0 = ok). */
static uint8_t measure_frame(uint32_t *transfer_ms)
{
    uint32_t t0;
    if (!IchiStampMeasure(ICHI_MEASURE_FRAME)) { return 3U; }
    IchiStampSelectClip(0U);
    t0 = tick_10ms;
    if (!IchiFeatureFrame(IchiStampReadPcm, FRAME_SAMPLES)) { return 4U; }
    *transfer_ms = (tick_10ms - t0) * 10UL;
    return 0U;
}

static float bf16_to_f(int16_t v)
{
    union { uint32_t bits; float value; } c;
    c.bits = ((uint32_t)(uint16_t)v) << 16;
    return c.value;
}

static float to_bf16(float v)                    /* round to nearest even, as the AxlCORE boundary */
{
    union { uint32_t bits; float value; } c;
    c.value = v;
    c.bits += 0x7FFFUL + ((c.bits >> 16) & 1UL);
    c.bits &= 0xFFFF0000UL;
    return c.value;
}

/* Factory ELM head on the CPU for the last ELM input (same arithmetic as the PC reference
   mcu_reference: bf16 alpha/beta, float accumulation, bf16 hidden and output). */
static uint8_t factory_predict(void)
{
    const int16_t *x = IchiInferenceElmInput();
    float h[ICHI_MODEL_HIDDEN_SIZE];
    float best = -1.0e30f;
    uint8_t j, i, k, cls = 0U;
    for (j = 0U; j < ICHI_MODEL_HIDDEN_SIZE; j++)
    {
        float z = 0.0f;
        for (i = 0U; i < ICHI_CAL_ALPHA_ROWS; i++)
        {
            z += bf16_to_f(x[i]) * bf16_to_f(ichi_calib_alpha[i * ICHI_MODEL_HIDDEN_SIZE + j]);
        }
        z = 0.2f * z + 0.5f;
        h[j] = to_bf16((z < 0.0f) ? 0.0f : ((z > 1.0f) ? 1.0f : z));
    }
    for (k = 0U; k < ICHI_MODEL_OUTPUT_SIZE; k++)
    {
        float o = 0.0f;
        for (j = 0U; j < ICHI_MODEL_HIDDEN_SIZE; j++)
        {
            o += h[j] * bf16_to_f(ichi_model_beta[j * ICHI_MODEL_OUTPUT_SIZE + k]);
        }
        o = to_bf16(o);
        if (o > best) { best = o; cls = k; }
    }
    return cls;
}

/* On-site calibration: returns 0 or the failed stage. */
static uint8_t calibrate(void)
{
    uint32_t start = tick_10ms;
    uint16_t correct = 0U, samples = 0U;
    uint8_t i, w, stage = 0U;
    IchiUiBanner("On-site calibration", ICHI_MAGENTA, ICHI_WHITE);
    (void)IchiInferenceCalibrationBegin();
    for (i = 0U; (i < SURVEY_STATES) && (stage == 0U) && !abort_request; i++)
    {
        uint8_t state = (uint8_t)(i ^ (i >> 1));
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
            uint32_t t0 = tick_10ms;
            uint8_t pred = 0U;
            IchiStampSelectClip(w);
            if (!IchiFeatureFrame(IchiStampReadPcm, FRAME_SAMPLES)) { stage = 4U; report_error(4U, state); break; }
            payload[4] = 0U;
            put_u32(&payload[4], (tick_10ms - t0) * 10UL);
            IchiFeatureInput(feature);
            (void)IchiInferenceRun((const uint8_t *)feature, output, &pred);
            if (!IchiInferenceTrain((const uint8_t *)feature, state)) { stage = 7U; report_error(7U, state); break; }
            samples++;
            if (pred == state) { correct++; }
            payload[0] = state;
            payload[1] = w;
            payload[2] = pred;
            payload[3] = 0U;
            send(MSG_CAL_STEP, 8U);
        }
    }
    if (abort_request || (stage != 0U)) { IchiInferenceUseFactory(); }
    put_u16(&payload[0], correct);
    put_u16(&payload[2], samples);
    payload[4] = (!abort_request && (stage == 0U)) ? 1U : 0U;
    put_u16(&payload[5], (tick_10ms - start) / 100UL);
    send(MSG_CAL_DONE, 7U);
    return stage;
}

static void run_survey(void)
{
    uint32_t start = tick_10ms;
    uint16_t correct = 0U, total = 0U, correct_factory = 0U;
    uint8_t round, i, stage = 0U;

    abort_request = false;
    payload[0] = SURVEY_ROUNDS;
    payload[1] = SURVEY_BASELINE;
    payload[2] = SURVEY_STATES;
    payload[3] = SURVEY_CALIBRATE;
    send(MSG_START, 4U);
    IchiUiShowState(ICHI_UI_NONE, ICHI_UI_NONE);
    IchiUiInfoLine("", ICHI_BLACK);

    IchiUiBanner("Closing all", ICHI_YELLOW, ICHI_BLACK);
    lcd("survey: close", "all servos");
    if (!drive_to(0U)) { report_error(6U, 0U); stage = 6U; }
    else { IchiUiShowState(0U, ICHI_UI_NONE); }

    IchiFeatureBaselineReset();
    for (i = 0U; (stage == 0U) && (i < SURVEY_BASELINE) && !abort_request; i++)
    {
        char text[] = "baseline 0/0";
        uint32_t ms = 0U;
        text[9] = (char)('1' + i);
        text[11] = (char)('0' + SURVEY_BASELINE);
        IchiUiBanner("Calibrating...", ICHI_CYAN, ICHI_BLACK);
        IchiUiInfoLine(text, ICHI_CYAN);
        lcd("survey baseline", &text[9]);
        stage = measure_frame(&ms);
        if (stage != 0U) { report_error(stage, 0U); break; }
        IchiFeatureBaselineAdd(SURVEY_BASELINE);
        payload[0] = i;
        payload[1] = IchiFeatureLastExponent();
        put_u32(&payload[2], ms);
        send(MSG_BASE, 6U);
    }

    if ((stage == 0U) && (SURVEY_CALIBRATE != 0U) && !abort_request)
    {
        stage = calibrate();
    }

    for (round = 0U; (stage == 0U) && (round < SURVEY_ROUNDS) && !abort_request; round++)
    {
        for (i = 0U; (i < SURVEY_STATES) && !abort_request; i++)
        {
            uint8_t k = (uint8_t)(((round & 1U) != 0U) ? (SURVEY_STATES - 1U - i) : i);
            uint8_t pos = (uint8_t)((k + round * 11U) % SURVEY_STATES);
            uint8_t state = (uint8_t)(pos ^ (pos >> 1));                  /* Gray code */
            float output[ICHI_INFERENCE_OUTPUT_COUNT];
            uint32_t transfer_ms = 0U, t1, t2, t3;
            uint8_t pred = 0U, pred_factory, k2;
            char text[] = "00/00 h00000";
            char line[] = "h00000>h00000 NG";
            char acc[] = "OK 000/000";

            put_dec(&text[0], (uint32_t)total + 1U, 2U);
            put_dec(&text[3], (uint32_t)SURVEY_ROUNDS * SURVEY_STATES, 2U);
            state_label(&text[6], state);
            lcd(&text[6], "moving / measure");

            IchiUiBanner("Moving...", ICHI_YELLOW, ICHI_BLACK);
            if (!drive_to(state)) { report_error(6U, state); stage = 6U; break; }
            if (abort_request) { break; }
            IchiUiShowState(state, ICHI_UI_NONE);
            IchiUiBanner("Listening...", ICHI_CYAN, ICHI_BLACK);
            stage = measure_frame(&transfer_ms);
            if (stage != 0U) { report_error(stage, state); break; }
            t1 = tick_10ms;
            IchiFeatureInput(feature);
            t2 = tick_10ms;
            if (!IchiInferenceRun((const uint8_t *)feature, output, &pred)) { stage = 5U; report_error(5U, state); break; }
            t3 = tick_10ms;
            pred_factory = factory_predict();

            total++;
            if (pred == state) { correct++; }
            if (pred_factory == state) { correct_factory++; }
            payload[0] = state;
            payload[1] = round;
            payload[2] = pred;
            payload[3] = pred_factory;
            put_u32(&payload[4], transfer_ms);
            put_u16(&payload[8], (t2 - t1) * 10UL);
            put_u16(&payload[10], (t3 - t2) * 10UL);
            for (k2 = 0U; k2 < ICHI_INFERENCE_OUTPUT_COUNT; k2++) { memcpy(&payload[12U + 4U * k2], &output[k2], 4U); }
            send(MSG_RESULT, (uint16_t)(12U + 4U * ICHI_INFERENCE_OUTPUT_COUNT));

            state_label(&line[0], state);
            state_label(&line[7], pred);
            if (pred == state) { line[14] = 'O'; line[15] = 'K'; }
            put_dec(&acc[3], correct, 3U);
            put_dec(&acc[7], total, 3U);
            IchiUiShowState(state, pred);                                   /* digits + verdict banner */
            {
                char info[] = "00/00 cal 00 fac 00";
                put_dec(&info[0], total, 2U);
                put_dec(&info[3], (uint32_t)SURVEY_ROUNDS * SURVEY_STATES, 2U);
                put_dec(&info[10], correct, 2U);
                put_dec(&info[17], correct_factory, 2U);
                IchiUiInfoLine(info, ICHI_WHITE);
            }
            lcd(line, acc);
        }
    }

    IchiServoEnable(false);
    put_u16(&payload[0], correct);
    put_u16(&payload[2], total);
    payload[4] = abort_request ? 1U : 0U;
    put_u16(&payload[5], (tick_10ms - start) / 100UL);
    put_u16(&payload[7], correct_factory);
    send(MSG_DONE, 9U);
    if (stage == 0U)
    {
        char info[] = "done 00/00  EXEC: rerun";
        put_dec(&info[5], correct, 2U);
        put_dec(&info[8], total, 2U);
        if (abort_request) { IchiUiBanner("Aborted", ICHI_RED, ICHI_WHITE); }
        IchiUiInfoLine(info, abort_request ? ICHI_RED : ICHI_YELLOW);
    }
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
    IchiUiBanner("Press EXEC: survey", ICHI_YELLOW, ICHI_BLACK);
    lcd("survey: press", "EXEC to start");
    Uart1StartReadByte(receive_byte);

    for (;;)
    {
        uint8_t sw;
        bool pressed;
        if (IchiStampGetSwitches(&sw, &pressed))
        {
            if (pressed && !exec_down)
            {
                exec_down = true;
                start_request = true;
            }
            else { exec_down = pressed; }
        }
        if (start_request)
        {
            start_request = false;
            run_survey();
            start_request = false;          /* ignore a G sent during the survey */
        }
        wait_ticks(1U);
    }
}

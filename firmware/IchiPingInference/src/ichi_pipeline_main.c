/*
 * Pipeline test main (replaces S_System/main.c; tools/build.ps1 -Main pipeline).
 *
 * The Stamp-S3A serves recorded PCM16 clips over I2C exactly as the product
 * will stream audio (firmware/StampPipelineTest/stamp).  For every clip the
 * Solist computes the N333 feature on-chip (ichi_feature.c), runs the int8
 * front-end + ELM head (ichi_inference.c) and reports over UART1:
 *   baseline clips -> baseline, then every state clip -> class.
 * Each state result is compared on-chip with the embedded self-test input of
 * the same class (generated from the same wav on the PC).
 *
 * UART (tools/pipeline_monitor.py), frames as in ichi_protocol.h:
 *   0x30 PIPE_INFO  clips u8 | baseline u8 | samples u16
 *   0x31 PIPE_CLIP  clip u8 | class u8 | pred u8 | pred_header u8 | mismatch u16 |
 *                   max_diff u8 | exponent u8 | transfer ms u32 | feature ms u16 |
 *                   retries u16 | outputs float32 x ICHI_INFERENCE_OUTPUT_COUNT (states only)
 *   0x32 PIPE_DONE  correct u8 | agree u8 | states u8 | ok u8 | total s u16 |
 *                   retries u32 | bytes u32
 *   0x33 PIPE_ERROR stage u8 | clip u8
 * SW2 starts another run.
 */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "Lcd.h"
#include "PeriodicHandler10ms.h"
#include "Regulator5VOutput.h"
#include "Sleep.h"
#include "SoftwareInterrupt.h"
#include "Sw.h"
#include "SystemError.h"
#include "SystemPowerControl.h"
#include "TimeControl.h"
#include "Uart1.h"
#include "clock.h"
#include "ichi_feature.h"
#include "ichi_inference.h"
#include "ichi_protocol.h"
#include "ichi_stamp_link.h"
#include "mcu.h"
#include "smpl_common.h"
#include "timer0_1.h"
#include "wdt.h"

#define MSG_PIPE_INFO  (0x30U)
#define MSG_PIPE_CLIP  (0x31U)
#define MSG_PIPE_DONE  (0x32U)
#define MSG_PIPE_ERROR (0x33U)
#define BASELINE_CLASS (0xFFU)

static volatile uint32_t tick_10ms;
static volatile bool transmit_busy;
static uint8_t transmit_buffer[ICHI_TX_PAYLOAD_CAPACITY + 48U];
static uint8_t payload[ICHI_TX_PAYLOAD_CAPACITY];
static int8_t feature[ICHI_FEAT_INPUTS];
static uint32_t sequence;

static void periodic_10ms(void)
{
    tick_10ms++;
}

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

static void draw(const char *line1, const char *line2)
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

static void put_dec2(char *d, uint32_t v) { d[0] = (char)('0' + (v / 10U) % 10U); d[1] = (char)('0' + v % 10U); }

static void report_error(uint8_t stage, uint8_t clip)
{
    char line2[17] = "stage 0 clip 00";
    payload[0] = stage;
    payload[1] = clip;
    send(MSG_PIPE_ERROR, 2U);
    line2[6] = (char)('0' + stage);
    put_dec2(&line2[13], clip);
    draw("Stamp link ERROR", line2);
}

static void run_pipeline(void)
{
    IchiStampInfo info;
    uint32_t run_start = tick_10ms;
    uint8_t correct = 0U, agree = 0U, states = 0U;
    uint8_t clip;
    bool ok = true;

    draw("Stamp pipeline", "connecting...");
    if (!IchiStampGetInfo(&info) || (info.baseline == 0U) || (info.clips <= info.baseline))
    {
        report_error(1U, 0U);
        return;
    }
    payload[0] = info.clips;
    payload[1] = info.baseline;
    put_u16(&payload[2], info.samples);
    send(MSG_PIPE_INFO, 4U);

    IchiFeatureBaselineReset();
    for (clip = 0U; ok && (clip < info.clips); clip++)
    {
        uint8_t class_id;
        uint32_t t0, t1, t2;
        char line1[17] = "Base 00/00";
        char line2[17] = "";

        if (!IchiStampGetClip(clip, &class_id)) { report_error(2U, clip); ok = false; break; }
        if (clip < info.baseline)
        {
            put_dec2(&line1[5], clip + 1U);
            put_dec2(&line1[8], info.baseline);
            draw(line1, "reading PCM...");
        }
        IchiStampSelectClip(clip);
        t0 = tick_10ms;
        if (!IchiFeatureFrame(IchiStampReadPcm, info.samples)) { report_error(3U, clip); ok = false; break; }
        t1 = tick_10ms;
        memset(payload, 0, 16U);
        payload[0] = clip;
        payload[1] = class_id;
        payload[7] = IchiFeatureLastExponent();
        put_u16(&payload[14], IchiStampRetryCount());
        if (clip < info.baseline)
        {
            IchiFeatureBaselineAdd(info.baseline);
            t2 = tick_10ms;
            put_u32(&payload[8], (t1 - t0) * 10UL);
            put_u16(&payload[12], (t2 - t1) * 10UL);
            send(MSG_PIPE_CLIP, 16U);
        }
        else
        {
            float output[ICHI_INFERENCE_OUTPUT_COUNT];
            float ref_output[ICHI_INFERENCE_OUTPUT_COUNT];
            uint8_t pred = 0xFFU, pred_header = 0xFFU, max_diff = 0U;
            uint16_t mismatch = 0U, k;

            IchiFeatureInput(feature);
            if (!IchiInferenceRun((const uint8_t *)feature, output, &pred)) { report_error(4U, clip); ok = false; break; }
            t2 = tick_10ms;
            if (class_id < ICHI_INFERENCE_CASE_COUNT)
            {
                const int8_t *ref = ichi_model_cases[class_id];
                for (k = 0U; k < ICHI_FEAT_INPUTS; k++)
                {
                    int16_t d = (int16_t)(feature[k] - ref[k]);
                    if (d < 0) { d = (int16_t)-d; }
                    if (d != 0) { mismatch++; }
                    if (d > max_diff) { max_diff = (uint8_t)d; }
                }
                (void)IchiInferenceSelfTest(class_id, ref_output, &pred_header);
            }
            states++;
            if (pred == class_id) { correct++; }
            if (pred == pred_header) { agree++; }
            payload[2] = pred;
            payload[3] = pred_header;
            put_u16(&payload[4], mismatch);
            payload[6] = max_diff;
            put_u32(&payload[8], (t1 - t0) * 10UL);
            put_u16(&payload[12], (t2 - t1) * 10UL);
            for (k = 0U; k < ICHI_INFERENCE_OUTPUT_COUNT; k++)
            {
                memcpy(&payload[16U + 4U * k], &output[k], 4U);
            }
            send(MSG_PIPE_CLIP, (uint16_t)(16U + 4U * ICHI_INFERENCE_OUTPUT_COUNT));

            /* "C13 P13 H13 d21" / "ok 05/06 ag 06" */
            memcpy(line1, "C00 P00 H00 d000", 17U);
            put_dec2(&line1[1], class_id);
            put_dec2(&line1[5], pred);
            put_dec2(&line1[9], pred_header);
            line1[13] = (char)('0' + (mismatch / 100U) % 10U);
            put_dec2(&line1[14], mismatch % 100U);
            memcpy(line2, "ok 00/00 ag 00", 15U);
            put_dec2(&line2[3], correct);
            put_dec2(&line2[6], states);
            put_dec2(&line2[12], agree);
            draw(line1, line2);
        }
    }
    payload[0] = correct;
    payload[1] = agree;
    payload[2] = states;
    payload[3] = ok ? 1U : 0U;
    put_u16(&payload[4], (tick_10ms - run_start) / 100UL);
    put_u32(&payload[6], IchiStampRetryCount());
    put_u32(&payload[10], IchiStampByteCount());
    send(MSG_PIPE_DONE, 14U);
    if (ok)
    {
        char line1[17] = "DONE ok 00/00";
        char line2[17] = "agree 00 SW2:re";
        put_dec2(&line1[8], correct);
        put_dec2(&line1[11], states);
        put_dec2(&line2[6], agree);
        draw(line1, line2);
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
    timer0_start();
    Uart1PeripheralInit();
    SystemErrorInit();
    SwInit();

    Regulator5VOutputInit();
    Regulator5VOutputOn();
    LcdPeripheralInit();
    LcdInit();
    LcdDisplayOnOff(LCD_DISPLAY_ON, LCD_CURSOR_OFF, LCD_CURSOR_BLINK_OFF);
    LcdClearDisplay();
    LcdBacklightOn();

    PeriodicHandler10msSetCallBack(periodic_10ms);
    SysTick->LOAD = 0x00FFFFFFUL;
    SysTick->VAL = 0UL;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_ENABLE_Msk;
    IchiInferenceInitialize();

    run_pipeline();
    for (;;)
    {
        if ((PORT5->P5DI & 0x01U) == 0U)     /* SW2 (active Low) */
        {
            run_pipeline();
        }
        wdt_clear();
        SleepChangetoHaltMode();
    }
}

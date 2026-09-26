/*
 * On-device learning (ODL / OS-ELM on the AxlCORE) check (tools/build.ps1 -Main ichi_odl_test_main).
 * Replaces S_System/main.c of the vendor project.
 *
 * Runs once, 3 s after reset:
 *   step 0: IchiInferenceCalibrationBegin() (factory beta + P0 of ichi_calib_prior.h), read back
 *   step 1..N: IchiInferenceTrain() on an embedded self-test case with its own class, read back
 * After every step beta and P are read with ODL_GetWeightBeta / ODL_GetWeightP and sent over
 * UART1 so tools/odl_check.py can compare them with the same OS-ELM update on the PC.
 *   0x60 ROW   matrix u8 (0 beta, 1 P) | step u8 | row u8 | 32 x bf16 (raw)
 *   0x61 STEP  step u8 | case u8 | class u8 | pred after u8 | outputs float32 x 32 |
 *              ELM input bf16 x ICHI_CAL_ALPHA_ROWS (the rest of the embedding is zero)
 *   0x62 DONE  steps u8
 */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define ODL_DISABLE_RAND_GENERATOR_ALPHA
#include "solistAi.h"

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
#include "ichi_inference.h"
#include "ichi_protocol.h"
#include "mcu.h"
#include "smpl_common.h"
#include "timer0_1.h"
#include "wdt.h"

#define MSG_ROW  (0x60U)
#define MSG_STEP (0x61U)
#define MSG_DONE (0x62U)
#define N        (ICHI_MODEL_HIDDEN_SIZE)

static volatile uint32_t tick_10ms;
static volatile bool transmit_busy;
static uint8_t transmit_buffer[ICHI_TX_PAYLOAD_CAPACITY + 48U];
static uint8_t payload[ICHI_TX_PAYLOAD_CAPACITY];
static uint32_t sequence;

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

static void dump(uint8_t step)
{
    uint8_t matrix, row;
    for (matrix = 0U; matrix < 2U; matrix++)
    {
        for (row = 0U; row < N; row++)
        {
            bfloat16 values[N];
            if (matrix == 0U)
            {
                ODL_GetWeightBeta(values, 0U, (uint32_t)row * ICHI_MODEL_OUTPUT_SIZE * 2U, ICHI_MODEL_OUTPUT_SIZE * 2U);
            }
            else
            {
                ODL_GetWeightP(values, 0U, (uint32_t)row * N * 2U, N * 2U);
            }
            payload[0] = matrix;
            payload[1] = step;
            payload[2] = row;
            memcpy(&payload[3], values, sizeof(values));
            send(MSG_ROW, (uint16_t)(3U + sizeof(values)));
        }
    }
}

int32_t main(void)
{
    static const uint8_t cases[] = { 0U, 5U, 10U, 20U, 31U, 0U, 0U, 0U };
    uint8_t step;
    uint32_t start;

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
    (void)LcdDraw(LCD_START_OF_FIRST_LINE, "ODL check       ");
    SysTick->LOAD = 0x00FFFFFFUL;
    SysTick->VAL = 0UL;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_ENABLE_Msk;

    start = tick_10ms;
    while ((tick_10ms - start) < 300U) { wdt_clear(); }        /* let the PC start listening */

    (void)IchiInferenceCalibrationBegin();
    dump(0U);
    for (step = 1U; step <= (uint8_t)sizeof(cases); step++)
    {
        uint8_t c = cases[step - 1U];
        uint8_t cls = ichi_model_case_class[c];
        float output[ICHI_INFERENCE_OUTPUT_COUNT];
        uint8_t pred = 0xFFU, k;
        (void)IchiInferenceTrain((const uint8_t *)ichi_model_cases[c], cls);
        memcpy(&payload[4U + 4U * ICHI_INFERENCE_OUTPUT_COUNT], IchiInferenceElmInput(), ICHI_CAL_ALPHA_ROWS * 2U);
        (void)IchiInferenceRun((const uint8_t *)ichi_model_cases[c], output, &pred);
        payload[0] = step;
        payload[1] = c;
        payload[2] = cls;
        payload[3] = pred;
        for (k = 0U; k < ICHI_INFERENCE_OUTPUT_COUNT; k++) { memcpy(&payload[4U + 4U * k], &output[k], 4U); }
        send(MSG_STEP, (uint16_t)(4U + 4U * ICHI_INFERENCE_OUTPUT_COUNT + ICHI_CAL_ALPHA_ROWS * 2U));
        dump(step);
    }
    payload[0] = (uint8_t)sizeof(cases);
    send(MSG_DONE, 1U);
    (void)LcdDraw(LCD_START_OF_FIRST_LINE, "ODL check done  ");
    for (;;) { wdt_clear(); }
}

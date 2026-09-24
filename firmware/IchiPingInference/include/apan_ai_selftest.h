#ifndef APAN_AI_SELFTEST_H
#define APAN_AI_SELFTEST_H

/* IchiPing Solist-AI inference check: 167-input / 32-hidden / 14-class ELM.
   Keeps the acrylic_pan collector interface names so the vendor-project overlay
   build (S_AcrylicPan) needs no makefile changes. */

#include <stdbool.h>
#include <stdint.h>

#include "ichiping_model.h"
#define APAN_AI_INPUT_COUNT  ((uint16_t)ICHI_MODEL_INPUT_SIZE)   /* 167 (raw features) or 14 (linear front-end scores) */
#define APAN_AI_OUTPUT_COUNT ((uint16_t)ICHI_MODEL_OUTPUT_SIZE)  /* 14 classes, or hidden size in alpha-probe mode */

void ApanAiSelfTestInitialize(void);
uint8_t ApanAiSelfTestCaseCount(void);
/* Embedded qualification vector (one per class). */
bool ApanAiSelfTestRun(uint8_t case_id, float output[APAN_AI_OUTPUT_COUNT],
                       uint8_t *class_id);
/* PC-supplied standardized feature vector, already scaled and bfloat16-encoded. */
bool ApanAiInfer(const int16_t input_bf16[APAN_AI_INPUT_COUNT],
                 float output[APAN_AI_OUTPUT_COUNT], uint8_t *class_id);
/* Accelerator time of the last successful prediction (SysTick ticks @48 MHz). */
uint32_t ApanAiLastPredictTicks(void);

#endif

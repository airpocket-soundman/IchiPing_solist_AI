#ifndef APAN_AI_SELFTEST_H
#define APAN_AI_SELFTEST_H

/* IchiPing Solist-AI inference check.  Keeps the acrylic_pan collector
   interface names so the vendor-project overlay build (S_AcrylicPan) needs no
   makefile changes.

   Two model types are generated into ichiping_model.h:
   - ELM only (sim/emit_board_model.py): AI_INFER carries ICHI_MODEL_INPUT_SIZE
     bfloat16 values that go straight to the accelerator.
   - CNN front-end + ELM head (sim/emit_frontend_model.py, ICHI_FRONTEND_ENABLED):
     AI_INFER carries ICHI_FRONT_INPUT_SIZE int8 values (quantized N333
     features); the Cortex-M0+ runs the int8 convolution front-end and the
     accelerator runs the ELM head on the embedding. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ichiping_model.h"
#define APAN_AI_INPUT_COUNT  ((uint16_t)ICHI_MODEL_INPUT_SIZE)   /* ELM accelerator input count */
#define APAN_AI_OUTPUT_COUNT ((uint16_t)ICHI_MODEL_OUTPUT_SIZE)  /* classes, or hidden size in alpha-probe mode */

#ifdef ICHI_FRONTEND_ENABLED
#define APAN_AI_INFER_BYTES  ((uint16_t)ICHI_FRONT_INPUT_SIZE)   /* int8 front-end input */
/* Scratch for two int8 activation buffers (ping-pong). */
#define APAN_AI_SCRATCH_BYTES ((size_t)(2U * ICHI_FRONT_MAX_ACT))
#else
#define APAN_AI_INFER_BYTES  ((uint16_t)(ICHI_MODEL_INPUT_SIZE * 2U))  /* bfloat16 ELM input */
#define APAN_AI_SCRATCH_BYTES ((size_t)0U)
#endif

void ApanAiSelfTestInitialize(void);
uint8_t ApanAiSelfTestCaseCount(void);
/* Embedded qualification vector (one per class).  scratch must hold
   APAN_AI_SCRATCH_BYTES (may be NULL for ELM-only models). */
bool ApanAiSelfTestRun(uint8_t case_id, uint8_t *scratch, size_t scratch_size,
                       float output[APAN_AI_OUTPUT_COUNT], uint8_t *class_id);
/* PC-supplied input of APAN_AI_INFER_BYTES bytes (bfloat16 ELM input, or int8
   front-end input when ICHI_FRONTEND_ENABLED). */
bool ApanAiInfer(const uint8_t *input, uint8_t *scratch, size_t scratch_size,
                 float output[APAN_AI_OUTPUT_COUNT], uint8_t *class_id);
/* Accelerator time of the last successful prediction (SysTick ticks @48 MHz). */
uint32_t ApanAiLastPredictTicks(void);

#endif

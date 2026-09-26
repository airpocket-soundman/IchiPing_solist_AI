#ifndef ICHI_INFERENCE_H
#define ICHI_INFERENCE_H

/* IchiPing inference on the ML63Q2557.

   The model is generated into ichiping_model.h:
   - CNN front-end + ELM head (sim/emit_frontend_model.py, ICHI_FRONTEND_ENABLED):
     input = ICHI_FRONT_INPUT_SIZE int8 (quantized N333 features).  The int8
     convolution front-end runs on the Cortex-M0+ and the ELM head on AxlCORE.
   - ELM only (sim/emit_board_model.py, also used for alpha probing):
     input = ICHI_MODEL_INPUT_SIZE bfloat16 values fed to AxlCORE directly. */

#include <stdbool.h>
#include <stdint.h>

#include "ichiping_model.h"

#ifdef ICHI_FRONTEND_ENABLED
#define ICHI_INFERENCE_INPUT_BYTES ((uint16_t)ICHI_FRONT_INPUT_SIZE)
#else
#define ICHI_INFERENCE_INPUT_BYTES ((uint16_t)(ICHI_MODEL_INPUT_SIZE * 2U))
#endif
#define ICHI_INFERENCE_OUTPUT_COUNT ((uint8_t)ICHI_MODEL_OUTPUT_SIZE)
#define ICHI_INFERENCE_CASE_COUNT   ((uint8_t)ICHI_MODEL_CASE_COUNT)

void IchiInferenceInitialize(void);
/* input must hold ICHI_INFERENCE_INPUT_BYTES bytes. */
bool IchiInferenceRun(const uint8_t *input, float output[ICHI_INFERENCE_OUTPUT_COUNT],
                      uint8_t *class_id);
/* Embedded qualification input (one per class). */
bool IchiInferenceSelfTest(uint8_t case_id, float output[ICHI_INFERENCE_OUTPUT_COUNT],
                           uint8_t *class_id);
/* On-site calibration (generated/ichi_calib_prior.h): loads the factory beta and
   P0 = w (G_f + lambda I)^-1, then every IchiInferenceTrain() runs one OS-ELM update on
   the AxlCORE (ODL_StartTrain).  The calibrated beta stays in the accelerator until
   IchiInferenceUseFactory() or a reset. */
bool IchiInferenceCalibrationBegin(void);
bool IchiInferenceTrain(const uint8_t *input, uint8_t class_id);
void IchiInferenceUseFactory(void);
/* bfloat16 ELM input of the last IchiInferenceRun/Train (front-end models only, else NULL). */
const int16_t *IchiInferenceElmInput(void);
/* AxlCORE time of the last prediction in microseconds (SysTick @48 MHz). */
uint32_t IchiInferenceLastAcceleratorUs(void);

#endif

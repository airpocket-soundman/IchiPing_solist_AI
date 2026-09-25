#ifndef ICHI_FEATURE_H
#define ICHI_FEATURE_H

/* N333 feature (noise_diff_norm, 400-3000 Hz, 334 bins) computed on the ML63Q2557.

   Same integer arithmetic as sim/board_fixed_feature.py (the PC reference):
     PCM16 frame -> 2048-point Hann (Q15), hop 1024
     -> real FFT 2048 = complex FFT 1024 on int16 block floating point + split
     -> power (float32) summed over the segments -> mean -> dB (floor -80)
     -> minus the int16 baseline (1/256 dB) -> normalized over all 1024 bins
     -> float16 rounding -> bins 50..383 standardized and quantized to int8.

   RAM: 4 KB FFT work (shared with the front-end activations, see
   ichi_shared_scratch), 4 KB power/dB, 2 KB baseline. */

#include <stdbool.h>
#include <stdint.h>

#define ICHI_FEAT_NFFT     (2048U)
#define ICHI_FEAT_HOP      (1024U)
#define ICHI_FEAT_BINS     (1024U)
#define ICHI_FEAT_INPUTS   (334U)

/* Copies count PCM16 samples starting at sample offset of the current frame
   into dst.  Returns false on a transfer error. */
typedef bool (*IchiPcmReader)(uint16_t offset, int16_t *dst, uint16_t count);

/* Clears the baseline (before the baseline frames). */
void IchiFeatureBaselineReset(void);
/* Power spectrum of one frame of total_samples samples into the internal dB
   buffer.  Returns false if the reader failed. */
bool IchiFeatureFrame(IchiPcmReader reader, uint16_t total_samples);
/* Adds the last frame to the baseline as one of frame_count baseline frames. */
void IchiFeatureBaselineAdd(uint8_t frame_count);
/* Quantized front-end input of the last frame (consumes the dB buffer). */
void IchiFeatureInput(int8_t out[ICHI_FEAT_INPUTS]);
/* Largest block-floating-point exponent of the last frame (diagnostics). */
uint8_t IchiFeatureLastExponent(void);

#endif

#ifndef ICHI_STAMP_LINK_H
#define ICHI_STAMP_LINK_H

/* I2C link to the Stamp-S3A (slave 0x42) on the shared I2CF0 bus (400 kHz).

   Every exchange is a write of a command followed by a read of its reply;
   the reply echoes the request so a stale reply is detected and retried.
     INFO  0x01              -> 'I' 'C' clips u8 baseline u8 samples u16 version u8 0
     CLIP  0x02 clip         -> clip u8 class u8 (0xFF = baseline) 0 0
     STATUS 0x03             -> 'S' switches u8 (bit0..4 = a b c AB BC, 1 = OPEN) exec u8 (1 = pressed) seq u8
     READ  0x10 clip off n   -> off u16 clip u8 n u8 | n x int16 PCM   (n <= 64)
     MEASURE 0x20 mode       -> 'M' state id 0   (starts a PRBS play + record; mode 0 one 2 s frame,
                                               mode 1 calibration: 6 s PRBS, windows 0..4 with 1 s hop)
     MSTAT  0x21             -> 'M' state id x   (state 0 idle 1 busy 2 done 3 error)
   (firmware/StampPipelineTest/stamp serves recorded clips; firmware/StampMeasure
    records live and serves the aligned frame as clip 0.)

   Polling master on I2CF0 (ML63Q2500 user's manual ch.20 flow).  The vendor
   LCD driver uses the same peripheral from its interrupt; the I2CF0 IRQ is
   masked during a transfer and its pending state cleared afterwards. */

#include <stdbool.h>
#include <stdint.h>

#define ICHI_STAMP_ADDR        (0x42U)
#define ICHI_STAMP_READ_MAX    (240U)     /* PCM samples per READ (StampMeasure; StampPipelineTest allows 64) */

typedef struct
{
    uint8_t clips;
    uint8_t baseline;
    uint16_t samples;
    uint8_t version;
} IchiStampInfo;

bool IchiStampGetInfo(IchiStampInfo *info);
bool IchiStampGetClip(uint8_t clip, uint8_t *class_id);
/* Door/window switches and EXEC read by the Stamp (firmware/StampMicTest implements STATUS). */
bool IchiStampGetSwitches(uint8_t *state, bool *exec_pressed);
/* Live measurement (StampMeasure): starts one PRBS play + record and waits until the
   aligned frame is ready (mode 0: about 4 s, read clip 0; mode 1: about 8 s, read
   clips 0..4 = the five calibration windows). */
#define ICHI_MEASURE_FRAME       (0U)
#define ICHI_MEASURE_CALIBRATION (1U)
bool IchiStampMeasure(uint8_t mode);
/* Selects the clip read by IchiStampReadPcm (an IchiPcmReader). */
void IchiStampSelectClip(uint8_t clip);
bool IchiStampReadPcm(uint16_t offset, int16_t *dst, uint16_t count);
/* Plain write to any device on the shared I2CF0 bus (e.g. PCA9685 0x40). */
bool IchiI2cWrite(uint8_t address, const uint8_t *data, uint8_t size);
/* Transfer statistics since boot. */
uint32_t IchiStampRetryCount(void);
uint32_t IchiStampByteCount(void);

#endif

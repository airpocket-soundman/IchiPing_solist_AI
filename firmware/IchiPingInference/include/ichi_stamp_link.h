#ifndef ICHI_STAMP_LINK_H
#define ICHI_STAMP_LINK_H

/* I2C link to the Stamp-S3A (slave 0x42) on the shared I2CF0 bus (400 kHz).

   Every exchange is a write of a command followed by a read of its reply;
   the reply echoes the request so a stale reply is detected and retried.
     INFO  0x01              -> 'I' 'C' clips u8 baseline u8 samples u16 version u8 0
     CLIP  0x02 clip         -> clip u8 class u8 (0xFF = baseline) 0 0
     READ  0x10 clip off n   -> off u16 clip u8 n u8 | n x int16 PCM   (n <= 64)
   (firmware/StampPipelineTest/stamp implements the Stamp side.)

   Polling master on I2CF0 (ML63Q2500 user's manual ch.20 flow).  The vendor
   LCD driver uses the same peripheral from its interrupt; the I2CF0 IRQ is
   masked during a transfer and its pending state cleared afterwards. */

#include <stdbool.h>
#include <stdint.h>

#define ICHI_STAMP_ADDR        (0x42U)
#define ICHI_STAMP_READ_MAX    (64U)      /* PCM samples per READ */

typedef struct
{
    uint8_t clips;
    uint8_t baseline;
    uint16_t samples;
    uint8_t version;
} IchiStampInfo;

bool IchiStampGetInfo(IchiStampInfo *info);
bool IchiStampGetClip(uint8_t clip, uint8_t *class_id);
/* Selects the clip read by IchiStampReadPcm (an IchiPcmReader). */
void IchiStampSelectClip(uint8_t clip);
bool IchiStampReadPcm(uint16_t offset, int16_t *dst, uint16_t count);
/* Transfer statistics since boot. */
uint32_t IchiStampRetryCount(void);
uint32_t IchiStampByteCount(void);

#endif

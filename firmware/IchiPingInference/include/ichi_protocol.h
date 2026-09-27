#ifndef ICHI_PROTOCOL_H
#define ICHI_PROTOCOL_H

/* Binary UART framing shared with tools/ichi_serial.py.

   raw frame = "APAN" | version u8 | type u8 | flags u16 | sequence u32 |
               timestamp u32 | payload_size u16 | payload | CRC-32 (IEEE, LE)
   wire      = COBS(raw frame) followed by a single 0x00 delimiter.
   (Same layout as the acrylic_pan collector, so existing captures and tools
   stay readable.) */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define ICHI_PROTOCOL_VERSION   (1U)

/* Message types are defined by each main (see the header comment of src/ichi_*_main.c). */

#define ICHI_RX_PAYLOAD_CAPACITY (352U)   /* >= 334 int8 front-end inputs / 167 bf16 ELM inputs */
#define ICHI_TX_PAYLOAD_CAPACITY (264U)   /* >= AI_RESULT with 64 float outputs */

typedef struct
{
    uint8_t message_type;
    uint32_t sequence;
    uint16_t payload_size;
    uint8_t payload[ICHI_RX_PAYLOAD_CAPACITY];
} IchiFrame;

typedef struct
{
    uint8_t encoded[ICHI_RX_PAYLOAD_CAPACITY + 48U];
    uint16_t encoded_size;
    uint32_t error_count;
} IchiDecoder;

void IchiDecoderInit(IchiDecoder *decoder);
/* Feed one received byte; returns true when a complete, valid frame is in *frame. */
bool IchiDecoderFeed(IchiDecoder *decoder, uint8_t byte, IchiFrame *frame);
/* Encode one frame including the trailing 0x00; returns bytes written or 0. */
size_t IchiEncodeFrame(uint8_t message_type, uint32_t sequence,
                       const uint8_t *payload, uint16_t payload_size,
                       uint8_t *encoded, size_t capacity);

#endif

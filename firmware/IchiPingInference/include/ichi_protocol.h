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

#define ICHI_MSG_HELLO          (0x01U)
#define ICHI_MSG_STATUS         (0x02U)
#define ICHI_MSG_AI_SELFTEST    (0x14U)   /* payload: case id u8 */
#define ICHI_MSG_AI_INFER       (0x16U)   /* payload: ICHI_INFERENCE_INPUT_BYTES */
#define ICHI_MSG_AI_RESULT      (0x21U)
#define ICHI_MSG_ACK            (0x70U)
#define ICHI_MSG_NACK           (0x71U)   /* payload: request type u8, reason u8 */

#define ICHI_NACK_BAD_REQUEST   (1U)
#define ICHI_NACK_UNSUPPORTED   (2U)
#define ICHI_NACK_INFERENCE     (3U)

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

#ifndef APAN_PROTOCOL_H
#define APAN_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#include "apan_capture.h"
#include "apan_inference.h"

#define APAN_PROTOCOL_VERSION (1U)
#define APAN_MESSAGE_HELLO      (0x01U)
#define APAN_MESSAGE_STATUS     (0x02U)
#define APAN_MESSAGE_START      (0x10U)
#define APAN_MESSAGE_STOP       (0x11U)
#define APAN_MESSAGE_SET_CONFIG (0x12U)
#define APAN_MESSAGE_CAPTURE    (0x13U)
#define APAN_MESSAGE_AI_SELFTEST (0x14U)
#define APAN_MESSAGE_SET_MODE    (0x15U)
#define APAN_MESSAGE_AI_INFER    (0x16U)   /* IchiPing: 167 bfloat16 ELM inputs, or 334 int8 CNN front-end inputs -> AI_RESULT */
#define APAN_MESSAGE_EVENT_DATA (0x20U)
#define APAN_MESSAGE_AI_RESULT  (0x21U)
#define APAN_MESSAGE_INFERENCE_EVENT (0x22U)
#define APAN_MESSAGE_EVENT_CHUNK (0x23U)
#define APAN_MESSAGE_POSITION_RESULT (0x24U)
#define APAN_MESSAGE_POSITION_DIAGNOSTIC (0x26U)
#define APAN_MESSAGE_ACK        (0x70U)
#define APAN_MESSAGE_NACK       (0x71U)
#define APAN_ENCODED_FRAME_CAPACITY (1120U)
#define APAN_COMMAND_PAYLOAD_CAPACITY (352U)  /* IchiPing AI_INFER: 167 x bfloat16 */
#define APAN_COMMAND_ENCODED_CAPACITY (400U)
#define APAN_MODE_COLLECT   (0U)
#define APAN_MODE_INFERENCE (1U)
#define APAN_MODE_INSTRUMENT (2U)
#define APAN_MODE_POSITION (3U)

typedef struct
{
    uint8_t message_type;
    uint16_t flags;
    uint32_t sequence;
    uint16_t payload_size;
    uint8_t payload[APAN_COMMAND_PAYLOAD_CAPACITY];
} ApanCommandFrame;

typedef struct
{
    uint8_t encoded[APAN_COMMAND_ENCODED_CAPACITY];
    uint16_t encoded_size;
    uint32_t error_count;
} ApanProtocolDecoder;

void ApanProtocolDecoderInit(ApanProtocolDecoder *decoder);
bool ApanProtocolDecoderFeed(ApanProtocolDecoder *decoder, uint8_t byte,
                             ApanCommandFrame *frame);

size_t ApanProtocolEncodeFrame(uint8_t message_type, uint16_t flags,
                               uint32_t sequence, uint32_t timestamp_us,
                               const uint8_t *payload, uint16_t payload_size,
                               uint8_t *encoded, size_t capacity);

/* Returns encoded bytes including the trailing zero delimiter, or zero on error. */
size_t ApanProtocolEncodeEvent(const ApanEvent *event,
                               uint32_t sequence,
                               uint32_t timestamp_us,
                               uint8_t *encoded,
                               size_t capacity);

/* One atomic live result: event metadata, class/scores, then 512 raw samples. */
size_t ApanProtocolEncodeInferenceEvent(const ApanEvent *event,
                                        uint8_t class_id,
                                        const float outputs[APAN_INFERENCE_OUTPUT_COUNT],
                                        uint32_t sequence,
                                        uint32_t timestamp_us,
                                        uint8_t *encoded,
                                        size_t capacity);

size_t ApanProtocolEncodeEventChunk(const ApanEvent *event,
                                    uint32_t event_id,
                                    uint16_t chunk_index,
                                    uint32_t sequence,
                                    uint32_t timestamp_us,
                                    uint8_t *encoded,
                                    size_t capacity);

#endif

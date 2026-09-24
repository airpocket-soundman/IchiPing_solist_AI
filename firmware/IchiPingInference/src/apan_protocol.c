#include "apan_protocol.h"

#include <stdbool.h>
#include <string.h>

#define RAW_HEADER_SIZE   (18U)
#define EVENT_HEADER_SIZE (12U)
#define CRC_SIZE          (4U)
#define EVENT_PAYLOAD_SIZE (EVENT_HEADER_SIZE + (APAN_EVENT_SAMPLES * 2U))
#define AI_RESULT_SIZE (4U + (APAN_INFERENCE_OUTPUT_COUNT * 4U))
#define EVENT_CHUNK_HEADER_SIZE (20U)
#define EVENT_CHUNK_SAMPLES (512U)
#define INFERENCE_PAYLOAD_SIZE (EVENT_HEADER_SIZE + AI_RESULT_SIZE + (APAN_EVENT_SAMPLES * 2U))
#define MAX_PAYLOAD_SIZE INFERENCE_PAYLOAD_SIZE
#define RAW_FRAME_SIZE (RAW_HEADER_SIZE + MAX_PAYLOAD_SIZE + CRC_SIZE)

static uint8_t raw_frame[RAW_FRAME_SIZE];

static void put_u16_le(uint8_t *target, uint16_t value)
{
    target[0] = (uint8_t)value;
    target[1] = (uint8_t)(value >> 8);
}

static void put_u32_le(uint8_t *target, uint32_t value)
{
    target[0] = (uint8_t)value;
    target[1] = (uint8_t)(value >> 8);
    target[2] = (uint8_t)(value >> 16);
    target[3] = (uint8_t)(value >> 24);
}

static uint16_t get_u16_le(const uint8_t *source)
{
    return (uint16_t)((uint16_t)source[0] | ((uint16_t)source[1] << 8));
}

static uint32_t get_u32_le(const uint8_t *source)
{
    return (uint32_t)source[0] | ((uint32_t)source[1] << 8) |
           ((uint32_t)source[2] << 16) | ((uint32_t)source[3] << 24);
}

static uint32_t crc32(const uint8_t *data, size_t size)
{
    uint32_t crc = 0xFFFFFFFFUL;
    size_t i;
    for (i = 0U; i < size; i++)
    {
        uint8_t bit;
        crc ^= data[i];
        for (bit = 0U; bit < 8U; bit++)
        {
            crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320UL : 0U);
        }
    }
    return crc ^ 0xFFFFFFFFUL;
}

static size_t cobs_encode(const uint8_t *input, size_t input_size,
                          uint8_t *output, size_t capacity)
{
    size_t read = 0U;
    size_t write = 1U;
    size_t code_index = 0U;
    uint8_t code = 1U;

    if (capacity < 2U)
    {
        return 0U;
    }
    while (read < input_size)
    {
        if (input[read] == 0U)
        {
            output[code_index] = code;
            code_index = write++;
            code = 1U;
            read++;
        }
        else
        {
            if (write >= capacity)
            {
                return 0U;
            }
            output[write++] = input[read++];
            code++;
            if (code == 0xFFU)
            {
                output[code_index] = code;
                code_index = write++;
                code = 1U;
            }
        }
        if (write >= capacity)
        {
            return 0U;
        }
    }
    output[code_index] = code;
    output[write++] = 0U;
    return write;
}

static size_t cobs_decode(const uint8_t *input, size_t input_size,
                          uint8_t *output, size_t capacity)
{
    size_t read = 0U;
    size_t write = 0U;
    while (read < input_size)
    {
        uint8_t code = input[read++];
        uint8_t i;
        if (code == 0U)
        {
            return 0U;
        }
        for (i = 1U; i < code; i++)
        {
            if ((read >= input_size) || (write >= capacity))
            {
                return 0U;
            }
            output[write++] = input[read++];
        }
        if ((code != 0xFFU) && (read < input_size))
        {
            if (write >= capacity)
            {
                return 0U;
            }
            output[write++] = 0U;
        }
    }
    return write;
}

void ApanProtocolDecoderInit(ApanProtocolDecoder *decoder)
{
    memset(decoder, 0, sizeof(*decoder));
}

bool ApanProtocolDecoderFeed(ApanProtocolDecoder *decoder, uint8_t byte,
                             ApanCommandFrame *frame)
{
    static uint8_t raw[RAW_HEADER_SIZE + APAN_COMMAND_PAYLOAD_CAPACITY + CRC_SIZE];
    size_t raw_size;
    uint16_t payload_size;
    uint32_t expected_crc;

    if (byte != 0U)
    {
        if (decoder->encoded_size < sizeof(decoder->encoded))
        {
            decoder->encoded[decoder->encoded_size++] = byte;
        }
        else
        {
            decoder->encoded_size = 0U;
            decoder->error_count++;
        }
        return false;
    }
    if (decoder->encoded_size == 0U)
    {
        return false;
    }
    raw_size = cobs_decode(decoder->encoded, decoder->encoded_size,
                           raw, sizeof(raw));
    decoder->encoded_size = 0U;
    if (raw_size < (RAW_HEADER_SIZE + CRC_SIZE))
    {
        decoder->error_count++;
        return false;
    }
    payload_size = get_u16_le(&raw[16]);
    if ((raw[0] != 'A') || (raw[1] != 'P') || (raw[2] != 'A') ||
        (raw[3] != 'N') || (raw[4] != APAN_PROTOCOL_VERSION) ||
        (payload_size > APAN_COMMAND_PAYLOAD_CAPACITY) ||
        (raw_size != (size_t)(RAW_HEADER_SIZE + payload_size + CRC_SIZE)))
    {
        decoder->error_count++;
        return false;
    }
    expected_crc = get_u32_le(&raw[raw_size - CRC_SIZE]);
    if (crc32(raw, raw_size - CRC_SIZE) != expected_crc)
    {
        decoder->error_count++;
        return false;
    }
    frame->message_type = raw[5];
    frame->flags = get_u16_le(&raw[6]);
    frame->sequence = get_u32_le(&raw[8]);
    frame->payload_size = payload_size;
    if (payload_size > 0U)
    {
        memcpy(frame->payload, &raw[RAW_HEADER_SIZE], payload_size);
    }
    return true;
}

size_t ApanProtocolEncodeFrame(uint8_t message_type, uint16_t flags,
                               uint32_t sequence, uint32_t timestamp_us,
                               const uint8_t *payload, uint16_t payload_size,
                               uint8_t *encoded, size_t capacity)
{
    uint32_t crc;
    size_t raw_size;
    if (payload_size > MAX_PAYLOAD_SIZE)
    {
        return 0U;
    }
    raw_frame[0] = 'A'; raw_frame[1] = 'P'; raw_frame[2] = 'A'; raw_frame[3] = 'N';
    raw_frame[4] = APAN_PROTOCOL_VERSION;
    raw_frame[5] = message_type;
    put_u16_le(&raw_frame[6], flags);
    put_u32_le(&raw_frame[8], sequence);
    put_u32_le(&raw_frame[12], timestamp_us);
    put_u16_le(&raw_frame[16], payload_size);
    if ((payload_size > 0U) && (payload != NULL))
    {
        memcpy(&raw_frame[RAW_HEADER_SIZE], payload, payload_size);
    }
    raw_size = RAW_HEADER_SIZE + payload_size;
    crc = crc32(raw_frame, raw_size);
    put_u32_le(&raw_frame[raw_size], crc);
    return cobs_encode(raw_frame, raw_size + CRC_SIZE, encoded, capacity);
}

size_t ApanProtocolEncodeInferenceEvent(const ApanEvent *event,
                                        uint8_t class_id,
                                        const float outputs[APAN_INFERENCE_OUTPUT_COUNT],
                                        uint32_t sequence,
                                        uint32_t timestamp_us,
                                        uint8_t *encoded,
                                        size_t capacity)
{
    uint8_t *payload = &raw_frame[RAW_HEADER_SIZE];
    uint16_t i;

    if ((event == NULL) || (event->sample_count != APAN_INFERENCE_SAMPLES))
    {
        return 0U;
    }

    put_u32_le(&payload[0], APAN_SAMPLE_RATE_HZ);
    put_u16_le(&payload[4], APAN_EVENT_SAMPLES);
    put_u16_le(&payload[6], event->trigger_index);
    put_u16_le(&payload[8], event->peak_abs);
    put_u16_le(&payload[10], 0U);
    payload[12] = 0xFFU;
    payload[13] = class_id;
    payload[14] = 0U;
    payload[15] = 0U;
    for (i = 0U; i < APAN_INFERENCE_OUTPUT_COUNT; i++)
    {
        union { float value; uint32_t bits; } packed;
        uint16_t offset = (uint16_t)(16U + (i * 4U));
        packed.value = outputs[i];
        put_u32_le(&payload[offset], packed.bits);
    }
    for (i = 0U; i < APAN_EVENT_SAMPLES; i++)
    {
        put_u16_le(&payload[EVENT_HEADER_SIZE + AI_RESULT_SIZE + (i * 2U)],
                   (uint16_t)event->samples[i]);
    }
    return ApanProtocolEncodeFrame(APAN_MESSAGE_INFERENCE_EVENT, 0U, sequence,
                                   timestamp_us, payload, INFERENCE_PAYLOAD_SIZE,
                                   encoded, capacity);
}

size_t ApanProtocolEncodeEvent(const ApanEvent *event,
                               uint32_t sequence,
                               uint32_t timestamp_us,
                               uint8_t *encoded,
                               size_t capacity)
{
    uint8_t *payload = &raw_frame[RAW_HEADER_SIZE];
    uint16_t i;
    uint16_t sample_count;

    if (event == NULL) { return 0U; }
    sample_count = event->sample_count;
    if ((sample_count == 0U) || (sample_count > APAN_EVENT_SAMPLES))
    {
        return 0U;
    }

    put_u32_le(&payload[0], APAN_SAMPLE_RATE_HZ);
    put_u16_le(&payload[4], sample_count);
    put_u16_le(&payload[6], event->trigger_index);
    put_u16_le(&payload[8], event->peak_abs);
    put_u16_le(&payload[10], 0U);
    for (i = 0U; i < sample_count; i++)
    {
        put_u16_le(&payload[EVENT_HEADER_SIZE + (i * 2U)], (uint16_t)event->samples[i]);
    }

    return ApanProtocolEncodeFrame(APAN_MESSAGE_EVENT_DATA, 0U, sequence,
                                   timestamp_us, payload,
                                   (uint16_t)(EVENT_HEADER_SIZE + sample_count * 2U),
                                   encoded, capacity);
}

size_t ApanProtocolEncodeEventChunk(const ApanEvent *event,
                                    uint32_t event_id,
                                    uint16_t chunk_index,
                                    uint32_t sequence,
                                    uint32_t timestamp_us,
                                    uint8_t *encoded,
                                    size_t capacity)
{
    uint8_t *payload = &raw_frame[RAW_HEADER_SIZE];
    uint16_t chunk_count;
    uint16_t chunk_samples;
    uint16_t offset;
    uint16_t i;

    if ((event == NULL) || (event->sample_count == 0U) ||
        (event->sample_count > APAN_EVENT_CAPACITY))
    {
        return 0U;
    }
    chunk_count = (uint16_t)((event->sample_count + EVENT_CHUNK_SAMPLES - 1U) /
                             EVENT_CHUNK_SAMPLES);
    if (chunk_index >= chunk_count) { return 0U; }
    offset = (uint16_t)(chunk_index * EVENT_CHUNK_SAMPLES);
    chunk_samples = (uint16_t)(event->sample_count - offset);
    if (chunk_samples > EVENT_CHUNK_SAMPLES) { chunk_samples = EVENT_CHUNK_SAMPLES; }

    put_u32_le(&payload[0], event_id);
    put_u16_le(&payload[4], chunk_index);
    put_u16_le(&payload[6], chunk_count);
    put_u32_le(&payload[8], APAN_SAMPLE_RATE_HZ);
    put_u16_le(&payload[12], event->sample_count);
    put_u16_le(&payload[14], event->trigger_index);
    put_u16_le(&payload[16], event->peak_abs);
    put_u16_le(&payload[18], chunk_samples);
    for (i = 0U; i < chunk_samples; i++)
    {
        put_u16_le(&payload[EVENT_CHUNK_HEADER_SIZE + i * 2U],
                   (uint16_t)event->samples[offset + i]);
    }
    return ApanProtocolEncodeFrame(
        APAN_MESSAGE_EVENT_CHUNK, 0U, sequence, timestamp_us, payload,
        (uint16_t)(EVENT_CHUNK_HEADER_SIZE + chunk_samples * 2U),
        encoded, capacity);
}

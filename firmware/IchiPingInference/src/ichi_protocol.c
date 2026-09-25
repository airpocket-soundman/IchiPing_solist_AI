#include "ichi_protocol.h"

#include <string.h>

#define HEADER_SIZE (18U)
#define CRC_SIZE    (4U)

static void put_u16(uint8_t *target, uint16_t value)
{
    target[0] = (uint8_t)value;
    target[1] = (uint8_t)(value >> 8);
}

static void put_u32(uint8_t *target, uint32_t value)
{
    target[0] = (uint8_t)value;
    target[1] = (uint8_t)(value >> 8);
    target[2] = (uint8_t)(value >> 16);
    target[3] = (uint8_t)(value >> 24);
}

static uint16_t get_u16(const uint8_t *source)
{
    return (uint16_t)((uint16_t)source[0] | ((uint16_t)source[1] << 8));
}

static uint32_t get_u32(const uint8_t *source)
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

void IchiDecoderInit(IchiDecoder *decoder)
{
    memset(decoder, 0, sizeof(*decoder));
}

bool IchiDecoderFeed(IchiDecoder *decoder, uint8_t byte, IchiFrame *frame)
{
    static uint8_t raw[HEADER_SIZE + ICHI_RX_PAYLOAD_CAPACITY + CRC_SIZE];
    size_t raw_size;
    uint16_t payload_size;

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
    raw_size = cobs_decode(decoder->encoded, decoder->encoded_size, raw, sizeof(raw));
    decoder->encoded_size = 0U;
    if (raw_size < (HEADER_SIZE + CRC_SIZE))
    {
        decoder->error_count++;
        return false;
    }
    payload_size = get_u16(&raw[16]);
    if ((memcmp(raw, "APAN", 4U) != 0) || (raw[4] != ICHI_PROTOCOL_VERSION) ||
        (payload_size > ICHI_RX_PAYLOAD_CAPACITY) ||
        (raw_size != (size_t)(HEADER_SIZE + payload_size + CRC_SIZE)) ||
        (crc32(raw, raw_size - CRC_SIZE) != get_u32(&raw[raw_size - CRC_SIZE])))
    {
        decoder->error_count++;
        return false;
    }
    frame->message_type = raw[5];
    frame->sequence = get_u32(&raw[8]);
    frame->payload_size = payload_size;
    if (payload_size > 0U)
    {
        memcpy(frame->payload, &raw[HEADER_SIZE], payload_size);
    }
    return true;
}

size_t IchiEncodeFrame(uint8_t message_type, uint32_t sequence,
                       const uint8_t *payload, uint16_t payload_size,
                       uint8_t *encoded, size_t capacity)
{
    static uint8_t raw[HEADER_SIZE + ICHI_TX_PAYLOAD_CAPACITY + CRC_SIZE];
    size_t raw_size;

    if (payload_size > ICHI_TX_PAYLOAD_CAPACITY)
    {
        return 0U;
    }
    memcpy(raw, "APAN", 4U);
    raw[4] = ICHI_PROTOCOL_VERSION;
    raw[5] = message_type;
    put_u16(&raw[6], 0U);
    put_u32(&raw[8], sequence);
    put_u32(&raw[12], 0U);
    put_u16(&raw[16], payload_size);
    if (payload_size > 0U)
    {
        memcpy(&raw[HEADER_SIZE], payload, payload_size);
    }
    raw_size = HEADER_SIZE + payload_size;
    put_u32(&raw[raw_size], crc32(raw, raw_size));
    return cobs_encode(raw, raw_size + CRC_SIZE, encoded, capacity);
}

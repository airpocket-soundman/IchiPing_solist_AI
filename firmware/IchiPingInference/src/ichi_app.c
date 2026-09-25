#include "ichi_app.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "Lcd.h"
#include "PeriodicHandler10ms.h"
#include "Uart1.h"
#include "ichi_inference.h"
#include "ichi_protocol.h"
#include "mcu.h"

/* Request/response over UART1 (115200 bps, tools/ichi_serial.py):
     HELLO       -> HELLO "IchiPing"
     STATUS      -> STATUS  version u8 | frontend u8 | input bytes u16 |
                            outputs u8 | self-test cases u8 | last total ms u16
     AI_SELFTEST -> AI_RESULT for embedded case <u8>
     AI_INFER    -> AI_RESULT for the supplied input
   AI_RESULT = case u8 (0xFE for AI_INFER) | class u8 | accelerator us u16 |
               total ms u16 | outputs float32[ICHI_INFERENCE_OUTPUT_COUNT] */

#define RESULT_HEADER_SIZE (6U)
#define INFER_CASE_ID      (0xFEU)

static IchiDecoder decoder;
static IchiFrame rx_frame;                 /* written by the UART receive interrupt */
static IchiFrame request;                  /* copy owned by the main loop */
static volatile bool request_pending;
static volatile bool transmit_busy;
static uint8_t transmit_buffer[ICHI_TX_PAYLOAD_CAPACITY + 48U];
static volatile uint32_t tick_10ms;
static bool lcd_ready;
static bool lcd_pending;
static uint8_t lcd_class_id;
static uint16_t lcd_total_ms;
static uint16_t last_total_ms;

static void receive_byte(uint32_t value, uint16_t error_status);

static void periodic_10ms(void)
{
    tick_10ms++;
}

static void transmit_complete(uint32_t count, uint16_t error_status)
{
    (void)count;
    (void)error_status;
    transmit_busy = false;
    /* Uart1Write replaces the interrupt-enable register with TX-only bits. */
    Uart1StartReadByte(receive_byte);
}

static void receive_byte(uint32_t value, uint16_t error_status)
{
    (void)error_status;
    if (IchiDecoderFeed(&decoder, (uint8_t)value, &rx_frame) && !request_pending)
    {
        memcpy(&request, &rx_frame, sizeof(request));
        request_pending = true;
    }
}

static void send(uint8_t type, uint32_t sequence, const uint8_t *payload, uint16_t size)
{
    size_t encoded = IchiEncodeFrame(type, sequence, payload, size,
                                     transmit_buffer, sizeof(transmit_buffer));
    if (encoded > 0U)
    {
        transmit_busy = true;
        Uart1Write(transmit_buffer, (uint32_t)encoded, transmit_complete);
    }
}

static void send_nack(const IchiFrame *frame, uint8_t reason)
{
    uint8_t payload[2];
    payload[0] = frame->message_type;
    payload[1] = reason;
    send(ICHI_MSG_NACK, frame->sequence, payload, 2U);
}

static uint16_t saturate_u16(uint32_t value)
{
    return (value > 0xFFFFUL) ? 0xFFFFU : (uint16_t)value;
}

static void put_u16(uint8_t *target, uint16_t value)
{
    target[0] = (uint8_t)value;
    target[1] = (uint8_t)(value >> 8);
}

static void handle_inference(const IchiFrame *frame)
{
    static uint8_t payload[RESULT_HEADER_SIZE + ICHI_INFERENCE_OUTPUT_COUNT * 4U];
    float output[ICHI_INFERENCE_OUTPUT_COUNT];
    uint8_t class_id;
    uint8_t case_id;
    uint8_t index;
    uint32_t start_tick = tick_10ms;
    bool ok;

    if (frame->message_type == ICHI_MSG_AI_SELFTEST)
    {
        case_id = (frame->payload_size > 0U) ? frame->payload[0] : 0U;
        ok = IchiInferenceSelfTest(case_id, output, &class_id);
    }
    else if (frame->payload_size == ICHI_INFERENCE_INPUT_BYTES)
    {
        case_id = INFER_CASE_ID;
        ok = IchiInferenceRun(frame->payload, output, &class_id);
    }
    else
    {
        send_nack(frame, ICHI_NACK_BAD_REQUEST);
        return;
    }
    if (!ok)
    {
        send_nack(frame, ICHI_NACK_INFERENCE);
        return;
    }
    last_total_ms = saturate_u16((tick_10ms - start_tick) * 10UL);
    payload[0] = case_id;
    payload[1] = class_id;
    put_u16(&payload[2], saturate_u16(IchiInferenceLastAcceleratorUs()));
    put_u16(&payload[4], last_total_ms);
    for (index = 0U; index < ICHI_INFERENCE_OUTPUT_COUNT; index++)
    {
        memcpy(&payload[RESULT_HEADER_SIZE + index * 4U], &output[index], 4U);   /* little endian */
    }
    send(ICHI_MSG_AI_RESULT, frame->sequence, payload, (uint16_t)sizeof(payload));
    lcd_class_id = class_id;
    lcd_total_ms = last_total_ms;
    lcd_pending = lcd_ready;
}

static void handle_request(const IchiFrame *frame)
{
    switch (frame->message_type)
    {
        case ICHI_MSG_HELLO:
        {
            static const uint8_t identity[] = "IchiPing";
            send(ICHI_MSG_HELLO, frame->sequence, identity, sizeof(identity) - 1U);
            break;
        }
        case ICHI_MSG_STATUS:
        {
            uint8_t payload[8];
#ifdef ICHI_FRONTEND_ENABLED
            payload[1] = 1U;
#else
            payload[1] = 0U;
#endif
            payload[0] = ICHI_PROTOCOL_VERSION;
            put_u16(&payload[2], ICHI_INFERENCE_INPUT_BYTES);
            payload[4] = ICHI_INFERENCE_OUTPUT_COUNT;
            payload[5] = ICHI_INFERENCE_CASE_COUNT;
            put_u16(&payload[6], last_total_ms);
            send(ICHI_MSG_STATUS, frame->sequence, payload, sizeof(payload));
            break;
        }
        case ICHI_MSG_AI_SELFTEST:
        case ICHI_MSG_AI_INFER:
            handle_inference(frame);
            break;
        default:
            send_nack(frame, ICHI_NACK_UNSUPPORTED);
            break;
    }
}

/* LCD: "S01101  cls 13  " / "T  1230ms       " (32 classes: door bits a b c AB BC). */
static void draw_result(void)
{
    char line1[17] = "S00000  cls 00  ";
    char line2[17] = "T 00000ms       ";
    uint8_t bit;
    uint16_t ms = lcd_total_ms;

    if (ICHI_INFERENCE_OUTPUT_COUNT == 32U)
    {
        for (bit = 0U; bit < 5U; bit++)
        {
            line1[1U + bit] = (char)('0' + ((lcd_class_id >> bit) & 1U));
        }
    }
    else
    {
        memcpy(line1, "      ", 6U);
    }
    line1[12] = (char)('0' + (lcd_class_id / 10U) % 10U);
    line1[13] = (char)('0' + lcd_class_id % 10U);
    for (bit = 0U; bit < 5U; bit++)
    {
        line2[6U - bit] = (char)('0' + ms % 10U);
        ms /= 10U;
    }
    (void)LcdDraw(LCD_START_OF_FIRST_LINE, line1);
    (void)LcdDraw(LCD_START_OF_SECOND_LINE, line2);
}

void IchiAppInitialize(bool lcd_available)
{
    lcd_ready = lcd_available;
    IchiDecoderInit(&decoder);
    IchiInferenceInitialize();
    PeriodicHandler10msSetCallBack(periodic_10ms);
    SysTick->LOAD = 0x00FFFFFFUL;
    SysTick->VAL = 0UL;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_ENABLE_Msk;
    Uart1StartReadByte(receive_byte);
}

void IchiAppProcess(void)
{
    if (transmit_busy)
    {
        return;
    }
    if (lcd_pending)
    {
        /* Draw after the result frame has left the UART. */
        lcd_pending = false;
        draw_result();
    }
    if (request_pending)
    {
        handle_request(&request);
        request_pending = false;
    }
}

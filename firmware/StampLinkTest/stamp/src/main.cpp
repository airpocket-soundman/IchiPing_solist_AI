// Stamp-S3A side of the Solist-AI <-> Stamp I2C link test.
//
// I2C slave 0x42 on the interposer bus (G13 = SDA, G15 = SCL, shared with the
// Solist LCD).  Every received frame is logged on USB CDC and flashes the
// on-board RGB LED; a read returns "STMP" + rx count (u16 LE) + last length.
#include <Arduino.h>
#include <Wire.h>
#include "driver/gpio.h"

static constexpr uint8_t kAddress = 0x42;
static constexpr int kSda = 13;
static constexpr int kScl = 15;
static constexpr int kRgbLed = 21;   // Stamp-S3A WS2812

static volatile uint16_t rxCount = 0;
static volatile uint8_t lastLen = 0;
static volatile bool rxPending = false;
static uint8_t lastData[32];

static void onReceive(int len)
{
    uint8_t n = 0;
    while (Wire.available()) {
        int b = Wire.read();
        if (n < sizeof(lastData)) lastData[n++] = (uint8_t)b;
    }
    lastLen = n;
    rxCount++;
    rxPending = true;
}

static void onRequest()
{
    uint16_t c = rxCount;
    uint8_t reply[7] = {'S', 'T', 'M', 'P', (uint8_t)c, (uint8_t)(c >> 8), lastLen};
    Wire.write(reply, sizeof(reply));
}

void setup()
{
    Serial.begin(115200);
    rgbLedWrite(kRgbLed, 0, 0, 8);
    bool ok = Wire.begin(kAddress, kSda, kScl, 100000);
    Wire.onReceive(onReceive);
    Wire.onRequest(onRequest);
    // Prime the first read reply.
    onRequest();
    Serial.printf("[stamp] I2C slave 0x%02X SDA=G%d SCL=G%d begin=%s\n", kAddress, kSda, kScl, ok ? "ok" : "FAIL");
}

void loop()
{
    static uint32_t lastBeat = 0;
    static uint32_t ledOffAt = 0;

    if (rxPending) {
        rxPending = false;
        uint8_t n = lastLen;
        Serial.printf("[stamp] rx #%u len=%u :", rxCount, n);
        for (uint8_t i = 0; i < n && i < sizeof(lastData); i++) Serial.printf(" %02X", lastData[i]);
        Serial.print("  \"");
        for (uint8_t i = 0; i < n && i < sizeof(lastData); i++) Serial.print(isprint(lastData[i]) ? (char)lastData[i] : '.');
        Serial.println("\"");
        rgbLedWrite(kRgbLed, 0, 24, 0);
        ledOffAt = millis() + 80;
    }
    if (ledOffAt != 0 && (int32_t)(millis() - ledOffAt) >= 0) {
        rgbLedWrite(kRgbLed, 0, 0, 8);
        ledOffAt = 0;
    }
    if (millis() - lastBeat >= 2000) {
        lastBeat = millis();
        Serial.printf("[stamp] alive rx=%u SDA=%d SCL=%d\n", rxCount, gpio_get_level((gpio_num_t)kSda), gpio_get_level((gpio_num_t)kScl));
    }
    delay(2);
}

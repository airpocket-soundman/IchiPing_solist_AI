// Stamp-S3A side of the Solist-AI pipeline test.
//
// Serves recorded PCM16 clips to the Solist over the interposer I2C bus
// (slave 0x42, G13 = SDA, G15 = SCL) the same way the product will stream the
// INMP441 audio.  Protocol (firmware/IchiPingInference/include/ichi_stamp_link.h):
//   write 0x01                -> read 'I' 'C' clips baseline samples(u16) version 0
//   write 0x02 clip           -> read clip class 0 0
//   write 0x10 clip off(u16) n -> read off(u16) clip n | n x PCM16 LE
// The reply is prepared when the command arrives and handed to the bus from
// onRequest (the ESP32-S3 slave stretches SCL until then).
#include <Arduino.h>
#include <Wire.h>

extern const uint8_t clips_start[] asm("_binary_data_clips_bin_start");

static constexpr uint8_t kAddress = 0x42;
static constexpr int kSda = 13;
static constexpr int kScl = 15;
static constexpr int kRgbLed = 21;
static constexpr uint8_t kReadMax = 64;

static uint8_t clipCount, baselineCount;
static uint16_t samplesPerClip;
static const uint8_t *classTable;
static const uint8_t *pcmBase;

static uint8_t reply[4 + 2 * kReadMax];
static volatile uint8_t replyLen = 0;
static volatile uint32_t readCount = 0, badCount = 0, requestCount = 0;
static volatile uint8_t lastClip = 0xFF;

static void onReceive(int len)
{
    uint8_t cmd[8];
    uint8_t n = 0;
    while (Wire.available()) {
        int b = Wire.read();
        if (n < sizeof(cmd)) cmd[n++] = (uint8_t)b;
    }
    replyLen = 0;
    if (n >= 1 && cmd[0] == 0x01) {
        reply[0] = 'I'; reply[1] = 'C';
        reply[2] = clipCount; reply[3] = baselineCount;
        reply[4] = (uint8_t)samplesPerClip; reply[5] = (uint8_t)(samplesPerClip >> 8);
        reply[6] = clips_start[4]; reply[7] = 0;
        replyLen = 8;
    } else if (n >= 2 && cmd[0] == 0x02 && cmd[1] < clipCount) {
        reply[0] = cmd[1]; reply[1] = classTable[cmd[1]]; reply[2] = 0; reply[3] = 0;
        replyLen = 4;
        lastClip = cmd[1];
    } else if (n >= 5 && cmd[0] == 0x10 && cmd[1] < clipCount && cmd[4] <= kReadMax) {
        uint16_t off = (uint16_t)(cmd[2] | (cmd[3] << 8));
        uint8_t cnt = cmd[4];
        if ((uint32_t)off + cnt <= samplesPerClip) {
            const uint8_t *src = pcmBase + ((uint32_t)cmd[1] * samplesPerClip + off) * 2U;
            reply[0] = cmd[2]; reply[1] = cmd[3]; reply[2] = cmd[1]; reply[3] = cnt;
            memcpy(&reply[4], src, 2U * cnt);
            replyLen = (uint8_t)(4 + 2 * cnt);
            readCount++;
        } else {
            badCount++;
        }
    } else {
        badCount++;
    }
}

static void onRequest()
{
    requestCount++;
    if (replyLen != 0) {
        Wire.write(reply, replyLen);
    } else {
        static const uint8_t none[4] = {0xEE, 0xEE, 0xEE, 0xEE};
        Wire.write(none, sizeof(none));
    }
}

void setup()
{
    Serial.begin(115200);
    rgbLedWrite(kRgbLed, 0, 0, 8);
    if (memcmp(clips_start, "ICLP", 4) != 0) {
        while (true) { Serial.println("[stamp] clips.bin missing"); delay(1000); }
    }
    clipCount = clips_start[5];
    baselineCount = clips_start[6];
    samplesPerClip = (uint16_t)(clips_start[8] | (clips_start[9] << 8));
    classTable = &clips_start[12];
    pcmBase = &clips_start[(12 + clipCount + 3) & ~3u];
    Wire.setBufferSize(sizeof(reply) + 8);
    bool ok = Wire.begin(kAddress, kSda, kScl, 400000);
    Wire.onReceive(onReceive);
    Wire.onRequest(onRequest);
    Serial.printf("[stamp] PCM server 0x%02X: %u clips (%u baseline) x %u samples, begin=%s\n",
                  kAddress, clipCount, baselineCount, samplesPerClip, ok ? "ok" : "FAIL");
}

void loop()
{
    static uint32_t lastLog = 0;
    static uint32_t lastReads = 0;
    static uint8_t shownClip = 0xFF;
    if (lastClip != shownClip) {
        shownClip = lastClip;
        uint8_t c = classTable[shownClip];
        Serial.printf("[stamp] clip %u (%s %u)\n", shownClip, c == 0xFF ? "baseline" : "class", c == 0xFF ? 0 : c);
        if (c == 0xFF) rgbLedWrite(kRgbLed, 0, 0, 24); else rgbLedWrite(kRgbLed, 0, 24, 0);
    }
    if (millis() - lastLog >= 5000) {
        lastLog = millis();
        uint32_t r = readCount;
        Serial.printf("[stamp] reads=%lu (+%lu/5s) requests=%lu bad=%lu\n",
                      (unsigned long)r, (unsigned long)(r - lastReads), (unsigned long)requestCount, (unsigned long)badCount);
        lastReads = r;
    }
    delay(5);
}

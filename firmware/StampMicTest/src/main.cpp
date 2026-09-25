// INMP441 microphone + MAX98357A amplifier bring-up on Stamp-S3A.
//
// One I2S port, master, 48 kHz, stereo, 32-bit slots (BCLK 3.072 MHz), Philips format.
// TX (amp DIN G7) and RX (mic SD G9) share BCLK G43 / WS G5 as in docs/io_allocation.md.
// INMP441 L/R = GND -> left slot.  MAX98357A SD_MODE = G1 (High = left channel, Low = shutdown);
// it is raised only after the clocks and zero PCM are running, and lowered first on stop.
// Speaker is 8 ohm 0.25 W with 12 dB amp gain, so the output level is limited to -20 dBFS.
//
// USB CDC commands (newline terminated):
//   P[dBFS]   play PRBS (16 kHz chips, default -41 dBFS = UNO Q amplitude 0.0088)
//   F<Hz>[ dBFS]  play a sine tone (default -41 dBFS)
//   S         stop output (SD_MODE Low, zero PCM)
//   A0 / A1   force amp SD_MODE Low / High (diagnostics)
//   D0..D3    drive strength of BCLK/WS/DOUT (diagnostics, default 0)
//   R<sec>    record mic: "PCM16 <n>\n" + int16 LE mono 16 kHz + "END\n" (output keeps running)
// Every 200 ms: "LVL out=<mode> rms=..dBFS peak=..dBFS dc=.. zero=.. right_rms=..dBFS"
#include <Arduino.h>
#include <math.h>
#include "driver/gpio.h"
#include "driver/i2s_std.h"

static constexpr int PIN_BCLK = 43;
static constexpr int PIN_WS = 5;
static constexpr int PIN_DOUT = 7;
static constexpr int PIN_DIN = 9;
static constexpr int PIN_AMP_SD = 1;
static constexpr uint32_t RATE = 48000;
static constexpr size_t FRAMES = 480;              // 10 ms per transfer
static constexpr float DEFAULT_DBFS = -41.1f;      // 0.0088 full scale
static constexpr float MAX_DBFS = -20.0f;

enum class Out { Off, Prbs, Tone };
static volatile Out out_mode = Out::Off;
static volatile float out_amp = 0.0f;              // linear, full scale = 1
static volatile float tone_hz = 1000.0f;

static i2s_chan_handle_t tx, rx;
static int32_t rx_buf[FRAMES * 2];

static void i2s_start()
{
    i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan.auto_clear = true;                                          // underrun -> zeros
    ESP_ERROR_CHECK(i2s_new_channel(&chan, &tx, &rx));
    i2s_std_config_t std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = (gpio_num_t)PIN_BCLK,
            .ws = (gpio_num_t)PIN_WS,
            .dout = (gpio_num_t)PIN_DOUT,
            .din = (gpio_num_t)PIN_DIN,
            .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(tx, &std));
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx, &std));
    ESP_ERROR_CHECK(i2s_channel_enable(tx));
    ESP_ERROR_CHECK(i2s_channel_enable(rx));
    // BCLK/WS are shared by the mic and the amp.  With the default drive (2) the lines
    // ring and the INMP441 miscounts clocks (-25 dBFS broadband garbage); with drive 3
    // the MAX98357A mutes.  Drive 0 gave clean mic data and normal playback (2026-09-25).
    for (int pin : {PIN_BCLK, PIN_WS, PIN_DOUT}) gpio_set_drive_capability((gpio_num_t)pin, GPIO_DRIVE_CAP_0);
}

// Continuous playback task: always writes (zeros when off) so BCLK/WS never stop.
// Runs on core 0 (Arduino loop/RX is on core 1); the tone uses a float table
// because the ESP32-S3 has no double-precision FPU.
static constexpr int SINE_N = 1024;
static float sine_table[SINE_N];

static void tx_task(void *)
{
    static int32_t buf[FRAMES * 2];
    uint32_t lfsr = 0xACE1u;
    int32_t chip = 0;
    uint32_t n = 0;
    uint32_t phase = 0;                                               // Q32 phase accumulator
    for (;;) {
        Out mode = out_mode;
        float amp = out_amp;
        for (size_t i = 0; i < FRAMES; i++, n++) {
            float v = 0.0f;
            if (mode == Out::Prbs) {
                if (n % 3 == 0) {                                     // 16 kHz chips
                    uint32_t bit = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1u;
                    lfsr = (lfsr >> 1) | (bit << 15);
                    chip = (lfsr & 1u) ? 1 : -1;
                }
                v = amp * (float)chip;
            } else if (mode == Out::Tone) {
                v = amp * sine_table[phase >> 22];
                phase += (uint32_t)(tone_hz * (4294967296.0f / RATE));
            }
            int32_t s = (int32_t)(v * 2147483392.0f);
            buf[2 * i] = s;
            buf[2 * i + 1] = s;
        }
        size_t written = 0;
        i2s_channel_write(tx, buf, sizeof(buf), &written, portMAX_DELAY);
    }
}

static void output_stop()
{
    digitalWrite(PIN_AMP_SD, LOW);                                   // amp off first
    out_mode = Out::Off;
    out_amp = 0.0f;
}

static void output_start(Out mode, float dbfs)
{
    dbfs = min(dbfs, MAX_DBFS);
    out_amp = powf(10.0f, dbfs / 20.0f);
    out_mode = mode;
    delay(50);                                                       // zero/stable PCM already flowing
    digitalWrite(PIN_AMP_SD, HIGH);                                  // High = left channel
}

static size_t read_frames()
{
    size_t bytes = 0;
    i2s_channel_read(rx, rx_buf, sizeof(rx_buf), &bytes, portMAX_DELAY);
    return bytes / 8;
}

static inline int32_t sample24(int32_t slot) { return slot >> 8; }
static float dbfs24(double v) { return v > 0 ? 20.0f * log10f((float)(v / 8388608.0)) : -140.0f; }

static void record(uint32_t seconds)
{
    const uint32_t total = seconds * 16000U;
    static int16_t out[FRAMES / 3];
    Serial.printf("PCM16 %lu\n", (unsigned long)total);
    uint32_t sent = 0;
    int64_t acc = 0;
    uint8_t phase = 0;
    while (sent < total) {
        size_t n = read_frames(), k = 0;
        for (size_t i = 0; i < n; i++) {
            acc += sample24(rx_buf[2 * i]);
            if (++phase == 3) {
                out[k++] = (int16_t)constrain((int32_t)(acc / 3) >> 8, -32768, 32767);
                acc = 0;
                phase = 0;
            }
        }
        size_t take = min<size_t>(k, total - sent);
        Serial.write((const uint8_t *)out, take * 2);
        sent += take;
    }
    Serial.print("END\n");
}

static void handle(const String &cmd)
{
    if (cmd.startsWith("R")) {
        record((uint32_t)max(1L, min(10L, cmd.substring(1).toInt())));
    } else if (cmd.startsWith("P")) {
        float db = cmd.length() > 1 ? cmd.substring(1).toFloat() : DEFAULT_DBFS;
        output_start(Out::Prbs, db);
        Serial.printf("OK PRBS %.1f dBFS\n", min(db, MAX_DBFS));
    } else if (cmd.startsWith("F")) {
        int sp = cmd.indexOf(' ');
        tone_hz = constrain(cmd.substring(1, sp < 0 ? cmd.length() : sp).toFloat(), 20.0f, 7000.0f);
        float db = sp < 0 ? DEFAULT_DBFS : cmd.substring(sp + 1).toFloat();
        output_start(Out::Tone, db);
        Serial.printf("OK TONE %.0f Hz %.1f dBFS\n", tone_hz, min(db, MAX_DBFS));
    } else if (cmd.startsWith("A")) {                                // A0/A1: force amp SD_MODE (diagnostics)
        digitalWrite(PIN_AMP_SD, cmd.substring(1).toInt() ? HIGH : LOW);
        Serial.printf("OK AMP_SD %d\n", digitalRead(PIN_AMP_SD));
    } else if (cmd.startsWith("D")) {                                // D0..D3: I2S output drive strength
        gpio_drive_cap_t cap = (gpio_drive_cap_t)constrain(cmd.substring(1).toInt(), 0, 3);
        for (int pin : {PIN_BCLK, PIN_WS, PIN_DOUT}) gpio_set_drive_capability((gpio_num_t)pin, cap);
        Serial.printf("OK DRIVE %d\n", (int)cap);
    } else if (cmd.startsWith("S")) {
        output_stop();
        Serial.println("OK STOP");
    }
}

void setup()
{
    pinMode(PIN_AMP_SD, OUTPUT);
    digitalWrite(PIN_AMP_SD, LOW);                                   // amplifier off during start-up
    Serial.begin(115200);
    i2s_start();
    for (int i = 0; i < SINE_N; i++) sine_table[i] = sinf(2.0f * (float)M_PI * i / SINE_N);
    xTaskCreatePinnedToCore(tx_task, "i2s_tx", 4096, nullptr, 5, nullptr, 0);
    for (int i = 0; i < 20; i++) read_frames();                      // discard start-up (>218 SCK cycles)
    Serial.println("INMP441 + MAX98357A test ready. P[dBFS] F<Hz>[ dBFS] S R<sec>");
}

void loop()
{
    static double sum2 = 0, sum = 0, sum2_r = 0;
    static int32_t peak = 0;
    static uint32_t count = 0, zero = 0, last = millis();
    static String cmd;

    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n') { handle(cmd); cmd = ""; }
        else if (c != '\r') { cmd += c; }
    }
    size_t n = read_frames();
    for (size_t i = 0; i < n; i++) {
        int32_t l = sample24(rx_buf[2 * i]), r = sample24(rx_buf[2 * i + 1]);
        sum += l;
        sum2 += (double)l * l;
        sum2_r += (double)r * r;
        peak = max(peak, abs(l));
        zero += (l == 0);
        count++;
    }
    if (millis() - last >= 200 && count > 0) {
        double mean = sum / count, var = sum2 / count - mean * mean;
        const char *mode = out_mode == Out::Prbs ? "prbs" : out_mode == Out::Tone ? "tone" : "off";
        Serial.printf("LVL out=%s rms=%.1fdBFS peak=%.1fdBFS dc=%ld zero=%.3f right_rms=%.1fdBFS\n",
                      mode, dbfs24(sqrt(var > 0 ? var : 0)), dbfs24(peak), (long)mean,
                      (double)zero / count, dbfs24(sqrt(sum2_r / count)));
        sum = sum2 = sum2_r = 0;
        peak = 0;
        count = zero = 0;
        last = millis();
    }
}

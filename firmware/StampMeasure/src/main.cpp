// Stamp-S3A measurement server for the Solist-AI (IchiPing).
//
// Audio (docs/io_allocation.md): one I2S port, master, 48 kHz, stereo, 32-bit slots,
// BCLK G43 / WS G5 shared by the INMP441 (SD -> G9, left slot) and the MAX98357A
// (DIN <- G7, SD_MODE G1).  BCLK/WS/DOUT drive strength CAP_0 (ringing otherwise).
//
// One measurement reproduces the UNO Q capture (IchiPing-UNO-Q uno_q/audio,
// app/python/ichiping_inference.py):
//   play 0.5 s silence + 2.0 s PRBS (data/prbs48k.bin, amplitude 0.0088) + 1.0 s silence,
//   record 3.0 s from the start, decimate 48 -> 16 kHz with the 63-tap Blackman FIR
//   (cutoff 0.15), locate the PRBS onset (energy rise, then +-200 sample correlation
//   with the +-1 chips), cut 32000 samples and scale x16 (collector word >> 12) to int16.
//
// I2C slave 0x42 (G13 SDA, G15 SCL, 400 kHz); write a command, then read the reply:
//   INFO    0x01              -> 'I' 'C' clips=1 baseline=0 samples u16 version 0
//   CLIP    0x02 0            -> 0 0xFE 0 0
//   STATUS  0x03              -> 'S' switches (bit0..4 a b c AB BC, 1 = OPEN) exec seq
//   READ    0x10 k off u16 n  -> off u16 k n | n x int16 PCM (n <= 240); window k starts
//                                 at onset + k x 16000 (k = 0 for a normal measurement)
//   MEASURE 0x20 [mode]       -> 'M' state id 0     (starts a measurement)
//     mode 0: 0.5 s lead + 2 s PRBS + 1 s tail, record 3 s  (one 2 s frame)
//     mode 1: 0.5 s lead + 3 x 2 s PRBS back to back + 0.5 s tail, record 7 s
//             (on-site calibration: five 2 s windows with 1 s hop, k = 0..4)
//     mode 2: no excitation, record 3 s (noise floor check); the frame starts at 0.5 s
//   MSTAT   0x21              -> 'M' state id onset_lo  (state 0 idle 1 busy 2 done 3 error)
// Switches: G44 a, G2 b, G4 c, G6 AB, G8 BC, G10 EXEC (INPUT_PULLUP, Low = CLOSE / pressed).
// USB CDC: one line per measurement ("MEAS id onset=... rms=...").  Commands (newline
// terminated): M = measure, C = calibration capture (7 s), Q = quiet capture (no PRBS), D = dump the aligned frame ("PCM16 32000" line + int16 LE + "END" line).
#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include "driver/gpio.h"
#include "driver/i2s_std.h"

extern const uint8_t prbs48k_start[] asm("_binary_data_prbs48k_bin_start");
extern const uint8_t chips_start[] asm("_binary_data_prbs_chips_bin_start");

static constexpr int PIN_BCLK = 43, PIN_WS = 5, PIN_DOUT = 7, PIN_DIN = 9, PIN_AMP_SD = 1;
static constexpr int PIN_SDA = 13, PIN_SCL = 15;
static constexpr int PIN_SW[5] = {44, 2, 4, 6, 8};
static constexpr int PIN_EXEC = 10;
static constexpr uint8_t I2C_ADDR = 0x42;
static constexpr uint32_t RATE = 48000;
static constexpr size_t BLOCK = 480;                     // 10 ms
static constexpr uint32_t LEAD = 24000, ACTIVE = 96000;                 // 48 kHz samples
static constexpr uint32_t CAPTURE_16K = 112000;          // up to 7.0 s at 16 kHz
static constexpr uint32_t WINDOW_HOP = 16000;            // calibration windows, 1 s
static constexpr uint32_t FRAME = 32000;                 // 2.0 s at 16 kHz
static constexpr int FIR_TAPS = 63;
static constexpr uint8_t READ_MAX = 240;

enum MeasState : uint8_t { MEAS_IDLE = 0, MEAS_BUSY = 1, MEAS_DONE = 2, MEAS_FAIL = 3 };

static i2s_chan_handle_t tx, rx;
static float fir[FIR_TAPS];
static int16_t cap16[CAPTURE_16K];                        // decimated capture, original int16 scale
static volatile uint32_t frame_start = 0;                 // index into cap16 of the aligned frame
static volatile uint8_t meas_state = MEAS_IDLE, meas_id = 0;
static volatile bool meas_request = false;
static volatile bool playing = false;                     // TX task plays the excitation
static volatile uint32_t play_pos = 0;
static volatile uint32_t play_active = ACTIVE, play_tail = 48000;  // current excitation layout
static volatile uint8_t meas_mode = 0;
static uint32_t cap_len16 = 0;                            // samples in cap16 of the last capture
static volatile uint8_t sw_stable = 0;
static uint8_t reply[4 + 2 * READ_MAX];
static volatile uint16_t reply_len = 0;              // up to 4 + 2 x READ_MAX = 484 (uint8_t cut it to 228)
static volatile uint8_t status_seq = 0;

// ---------------------------------------------------------------- I2S
static void i2s_start()
{
    i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan.auto_clear = true;
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
    for (int pin : {PIN_BCLK, PIN_WS, PIN_DOUT}) gpio_set_drive_capability((gpio_num_t)pin, GPIO_DRIVE_CAP_0);
}

// Always writes (zeros when idle) so BCLK/WS never stop.  During a measurement it plays
// lead silence, the PRBS and tail silence from play_pos (48 kHz samples).
static void tx_task(void *)
{
    static int32_t buf[BLOCK * 2];
    const int32_t *prbs = (const int32_t *)prbs48k_start;
    for (;;) {
        for (size_t i = 0; i < BLOCK; i++) {
            int32_t s = 0;
            if (playing) {
                uint32_t p = play_pos++;
                if (p >= LEAD && p < LEAD + play_active) s = prbs[(p - LEAD) % ACTIVE];
                if (p >= LEAD + play_active + play_tail) playing = false;
            }
            buf[2 * i] = s;
            buf[2 * i + 1] = s;
        }
        size_t written = 0;
        i2s_channel_write(tx, buf, sizeof(buf), &written, portMAX_DELAY);
    }
}

// ---------------------------------------------------------------- measurement
static void make_fir()
{
    // IchiPing-UNO-Q decimate_48k_to_16k: 63 taps, cutoff 0.15, Blackman, normalized.
    double sum = 0.0, k[FIR_TAPS];
    for (int i = 0; i < FIR_TAPS; i++) {
        double n = i - (FIR_TAPS - 1) / 2.0;
        double x = 2.0 * 0.15 * n;
        double sinc = (n == 0.0) ? 1.0 : sin(M_PI * x) / (M_PI * x);
        double w = 0.42 - 0.5 * cos(2.0 * M_PI * i / (FIR_TAPS - 1)) + 0.08 * cos(4.0 * M_PI * i / (FIR_TAPS - 1));
        k[i] = 2.0 * 0.15 * sinc * w;
        sum += k[i];
    }
    for (int i = 0; i < FIR_TAPS; i++) fir[i] = (float)(k[i] / sum);
}

static inline int chip(uint32_t i) { return ((chips_start[i >> 3] >> (i & 7)) & 1u) ? 1 : -1; }

// Records capture_48k samples while the excitation plays; returns false on an I2S error.
static bool capture(uint32_t capture_48k)
{
    static int32_t buf[BLOCK * 2];
    float ring[FIR_TAPS] = {0};
    uint32_t in = 0, out = 0;
    int head = 0;

    i2s_channel_read(rx, buf, sizeof(buf), nullptr, 100);            // drop a stale block
    play_pos = 0;
    playing = true;
    while (in < capture_48k) {
        size_t bytes = 0;
        if (i2s_channel_read(rx, buf, sizeof(buf), &bytes, 1000) != ESP_OK) return false;
        for (size_t i = 0; i < bytes / 8 && in < capture_48k; i++, in++) {
            ring[head] = (float)buf[2 * i] * (1.0f / 2147483648.0f);   // left slot, word full scale
            head = (head + 1) % FIR_TAPS;
            // "same" convolution: output n uses inputs n-31..n+31 -> ready at input n+31.
            if (in >= 31 && ((in - 31) % 3) == 0 && out < CAPTURE_16K) {
                float acc = 0.0f;
                int idx = head;                                         // oldest sample
                for (int t = FIR_TAPS - 1; t >= 0; t--) { acc += ring[idx] * fir[t]; idx = (idx + 1) % FIR_TAPS; }
                float y = floorf(acc * 16.0f * 32768.0f);               // x16 original scale
                cap16[out++] = (int16_t)(y > 32767.0f ? 32767 : (y < -32768.0f ? -32768 : y));
            }
        }
    }
    cap_len16 = out;
    return true;
}

// Onset search.  Playback and capture share the I2S clock, so the PRBS onset is almost
// always at EXPECTED_ONSET (0.542 s: 0.5 s lead silence + output/acoustic latency).  An
// energy threshold was fooled by low-frequency room thumps in the lead silence (2 of 5
// frames aligned 0.14-0.21 s early on 2026-09-26), so the onset is found by correlation
// only: a coarse search (every 4th lag, first 8000 chips) over +-0.2 s around the expected
// onset, then the full 32000-chip correlation over +-8 samples.
static constexpr int32_t EXPECTED_ONSET = 8673;
static constexpr int32_t SEARCH = 3200;

static int64_t corr(uint32_t lag, uint32_t n)
{
    int64_t c = 0;                                                      // sum x * chip
    for (uint32_t i = 0; i < n; i++) c += chip(i) > 0 ? cap16[lag + i] : -cap16[lag + i];
    return c;
}

static bool align(uint32_t needed, uint32_t *onset, double *rms_out)
{
    int64_t total = 0;
    for (uint32_t i = 0; i < cap_len16; i++) total += cap16[i];
    const double mean = (double)total / cap_len16;
    int64_t chip_sum_short = 0, chip_sum = 0;
    for (uint32_t i = 0; i < FRAME; i++) { chip_sum += chip(i); if (i < 8000) chip_sum_short += chip(i); }
    int32_t lo = EXPECTED_ONSET - SEARCH, hi = EXPECTED_ONSET + SEARCH;
    if (lo < 0) lo = 0;
    if (hi > (int32_t)(cap_len16 - needed)) hi = (int32_t)(cap_len16 - needed);
    if (hi < lo) return false;
    double best = -1.0;
    int32_t coarse = EXPECTED_ONSET;
    for (int32_t lag = lo; lag <= hi; lag += 4) {
        double c = fabs((double)corr((uint32_t)lag, 8000) - mean * (double)chip_sum_short);
        if (c > best) { best = c; coarse = lag; }
    }
    best = -1.0;
    uint32_t best_lag = (uint32_t)coarse;
    for (int32_t lag = coarse - 8; lag <= coarse + 8; lag++) {
        if (lag < lo || lag > hi) continue;
        double c = fabs((double)corr((uint32_t)lag, FRAME) - mean * (double)chip_sum);
        if (c > best) { best = c; best_lag = (uint32_t)lag; }
    }
    double e = 0.0;
    for (uint32_t i = 0; i < FRAME; i++) { double v = cap16[best_lag + i] - mean; e += v * v; }
    *onset = best_lag;
    *rms_out = sqrt(e / FRAME);
    return true;
}

static void measure()
{
    const bool cal = (meas_mode == 1);
    const bool quiet = (meas_mode == 2);
    play_active = cal ? 3 * ACTIVE : ACTIVE;
    play_tail = cal ? 24000 : 48000;
    const uint32_t capture_48k = cal ? 336000 : 144000;                // 7 s / 3 s
    const uint32_t needed = cal ? FRAME + 4 * WINDOW_HOP : FRAME;      // samples after the onset
    if (!quiet) digitalWrite(PIN_AMP_SD, HIGH);                        // zero PCM already flowing
    if (quiet) play_active = 0;                                         // silence only
    bool ok = capture(capture_48k);
    digitalWrite(PIN_AMP_SD, LOW);
    uint32_t onset = 0;
    double rms = 0.0;
    if (quiet) {
        onset = 8000;                                                   // fixed 0.5 s, no PRBS to align on
        double e = 0.0, m = 0.0;
        for (uint32_t i = 0; i < FRAME; i++) m += cap16[onset + i];
        m /= FRAME;
        for (uint32_t i = 0; i < FRAME; i++) { double v = cap16[onset + i] - m; e += v * v; }
        rms = sqrt(e / FRAME);
    } else {
        ok = ok && align(needed, &onset, &rms);
    }
    frame_start = onset;
    meas_state = ok ? MEAS_DONE : MEAS_FAIL;
    Serial.printf("MEAS %u mode %u %s onset=%lu (%.3f s) rms=%.0f\n", meas_id, meas_mode, ok ? "ok" : "FAIL",
                  (unsigned long)onset, onset / 16000.0, rms);
}

// ---------------------------------------------------------------- switches
static uint8_t read_switches()
{
    uint8_t v = 0;
    for (int i = 0; i < 5; i++) v |= (uint8_t)(digitalRead(PIN_SW[i]) == HIGH ? 1u : 0u) << i;
    v |= (uint8_t)(digitalRead(PIN_EXEC) == LOW ? 1u : 0u) << 5;
    return v;
}

static void switch_task(void *)
{
    uint8_t candidate = read_switches();
    uint32_t since = millis();
    for (;;) {
        uint8_t now = read_switches();
        if (now != candidate) { candidate = now; since = millis(); }
        else if (candidate != sw_stable && millis() - since >= 20) sw_stable = candidate;
        vTaskDelay(1);
    }
}

// ---------------------------------------------------------------- I2C slave
static void on_receive(int)
{
    uint8_t cmd[8];
    uint8_t n = 0;
    while (Wire.available()) { int b = Wire.read(); if (n < sizeof(cmd)) cmd[n++] = (uint8_t)b; }
    reply_len = 0;
    if (n == 0) return;
    switch (cmd[0]) {
        case 0x01:
            reply[0] = 'I'; reply[1] = 'C'; reply[2] = 1; reply[3] = 0;
            reply[4] = (uint8_t)FRAME; reply[5] = (uint8_t)(FRAME >> 8); reply[6] = 2; reply[7] = 0;
            reply_len = 8;
            break;
        case 0x02:
            reply[0] = 0; reply[1] = 0xFE; reply[2] = 0; reply[3] = 0;
            reply_len = 4;
            break;
        case 0x03: {
            uint8_t v = sw_stable;
            reply[0] = 'S'; reply[1] = v & 0x1F; reply[2] = (v >> 5) & 1; reply[3] = status_seq++;
            reply_len = 4;
            break;
        }
        case 0x10:
            if (n >= 5 && cmd[4] <= READ_MAX && meas_state == MEAS_DONE && cmd[1] <= 4) {
                uint16_t off = (uint16_t)(cmd[2] | (cmd[3] << 8));
                uint8_t cnt = cmd[4];
                uint32_t base = frame_start + (uint32_t)cmd[1] * WINDOW_HOP;
                if ((uint32_t)off + cnt <= FRAME && base + off + cnt <= cap_len16) {
                    reply[0] = cmd[2]; reply[1] = cmd[3]; reply[2] = cmd[1]; reply[3] = cnt;
                    memcpy(&reply[4], &cap16[base + off], 2u * cnt);
                    reply_len = (uint16_t)(4 + 2 * cnt);
                }
            }
            break;
        case 0x20:
            if (meas_state != MEAS_BUSY) {
                meas_id++;
                meas_mode = (n >= 2 && cmd[1] == 1) ? 1 : 0;
                meas_state = MEAS_BUSY;
                meas_request = true;
            }
            reply[0] = 'M'; reply[1] = meas_state; reply[2] = meas_id; reply[3] = 0;
            reply_len = 4;
            break;
        case 0x21:
            reply[0] = 'M'; reply[1] = meas_state; reply[2] = meas_id; reply[3] = (uint8_t)frame_start;
            reply_len = 4;
            break;
        default:
            break;
    }
}

static void on_request()
{
    static const uint8_t none[4] = {0xEE, 0xEE, 0xEE, 0xEE};
    if (reply_len != 0) Wire.write(reply, reply_len);
    else Wire.write(none, sizeof(none));
}

// ---------------------------------------------------------------- setup / loop
void setup()
{
    pinMode(PIN_AMP_SD, OUTPUT);
    digitalWrite(PIN_AMP_SD, LOW);                                      // amplifier off
    for (int pin : PIN_SW) pinMode(pin, INPUT_PULLUP);
    pinMode(PIN_EXEC, INPUT_PULLUP);
    sw_stable = read_switches();
    Serial.begin(115200);
    Serial.setTxTimeoutMs(0);
    make_fir();
    i2s_start();
    xTaskCreatePinnedToCore(tx_task, "i2s_tx", 4096, nullptr, 5, nullptr, 0);
    xTaskCreatePinnedToCore(switch_task, "switches", 3072, nullptr, 4, nullptr, 0);
    Wire.setBufferSize(sizeof(reply) + 8);
    Wire.begin(I2C_ADDR, PIN_SDA, PIN_SCL, 400000);
    Wire.onReceive(on_receive);
    Wire.onRequest(on_request);
    Serial.println("IchiPing Stamp measurement server ready (I2C 0x42)");
}

static void usb_commands()
{
    static String cmd;
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c != '\n') { if (c != '\r') cmd += c; continue; }
        if ((cmd == "M" || cmd == "C" || cmd == "Q") && meas_state != MEAS_BUSY) {
            meas_id++; meas_mode = (cmd == "C") ? 1 : ((cmd == "Q") ? 2 : 0); meas_state = MEAS_BUSY; meas_request = true;
        }
        else if (cmd == "D" && meas_state == MEAS_DONE) {
            Serial.setTxTimeoutMs(1000);
            Serial.printf("PCM16 %lu\n", (unsigned long)FRAME);
            Serial.write((const uint8_t *)&cap16[frame_start], 2u * FRAME);
            Serial.print("END\n");
            Serial.setTxTimeoutMs(0);
        }
        cmd = "";
    }
}

void loop()
{
    usb_commands();
    if (meas_request) {
        meas_request = false;
        measure();
    }
    // Keep the RX DMA drained while idle so a measurement starts with fresh data.
    static int32_t drain[BLOCK * 2];
    size_t bytes = 0;
    i2s_channel_read(rx, drain, sizeof(drain), &bytes, 20);
}

#include "ichi_feature.h"

#include <math.h>

#include "ichi_feature_tables.h"
#include "ichiping_model.h"
#include "wdt.h"

#define NC         (1024U)          /* complex FFT length */
#define BASE_SCALE (256.0f)         /* baseline dB fixed point */

/* 4 KB work area: int16 re/im interleaved FFT buffer during feature
   extraction, int8 activations during the front-end (ichi_inference.c). */
int8_t ichi_shared_scratch[ICHI_FEAT_NFFT * 2U] __attribute__((aligned(4)));

static float power_db[ICHI_FEAT_BINS];      /* power sum -> mean -> dB -> normalized */
static int16_t baseline_q[ICHI_FEAT_BINS];  /* 1/256 dB */
static uint8_t last_exponent;

static int32_t cos_q(uint32_t k)            /* cos(2 pi k / 2048) x 32767 */
{
    k &= (ICHI_FEAT_NFFT - 1U);
    return (k < NC) ? (int32_t)ichi_cos_q15[k] : -(int32_t)ichi_cos_q15[k - NC];
}

static int32_t sin_q(uint32_t k)            /* sin(2 pi k / 2048) = cos(2 pi (k - 512) / 2048) */
{
    return cos_q(k + ICHI_FEAT_NFFT - ICHI_FEAT_NFFT / 4U);
}

static int32_t shift_round(int32_t v, uint8_t s)
{
    return (s == 0U) ? v : ((v + (1L << (s - 1U))) >> s);
}

static int32_t round_half_even(float v)
{
    float f = floorf(v);
    float r = v - f;
    int32_t i = (int32_t)f;
    if (r > 0.5f) { return i + 1; }
    if (r < 0.5f) { return i; }
    return ((i & 1L) != 0L) ? i + 1 : i;
}

/* float32 -> float16 -> float32 (round half to even), like numpy astype(float16). */
static float round_float16(float v)
{
    int e;
    int q;
    if (v == 0.0f) { return 0.0f; }
    (void)frexpf(fabsf(v), &e);             /* |v| = m 2^e, m in [0.5, 1) */
    q = e - 1 - 10;                          /* quantum of a 10-bit mantissa */
    if (q < -24) { q = -24; }                /* float16 subnormal step */
    return ldexpf((float)round_half_even(ldexpf(v, -q)), q);
}

static uint16_t bit_reverse10(uint16_t v)
{
    uint16_t r = 0U;
    uint8_t i;
    for (i = 0U; i < 10U; i++)
    {
        r = (uint16_t)((r << 1) | (v & 1U));
        v >>= 1;
    }
    return r;
}

/* In-place radix-2 DIT complex FFT, int16 block floating point. */
static uint8_t fft1024(int16_t *z)
{
    uint16_t i, half;
    uint8_t exponent = 0U;

    for (i = 0U; i < NC; i++)
    {
        uint16_t j = bit_reverse10(i);
        if (j > i)
        {
            int16_t tr = z[2U * i], ti = z[2U * i + 1U];
            z[2U * i] = z[2U * j];
            z[2U * i + 1U] = z[2U * j + 1U];
            z[2U * j] = tr;
            z[2U * j + 1U] = ti;
        }
    }
    for (half = 1U; half < NC; half = (uint16_t)(half * 2U))
    {
        int32_t peak = 0;
        uint8_t s;
        uint16_t k;
        for (i = 0U; i < 2U * NC; i++)
        {
            int32_t a = (z[i] < 0) ? -(int32_t)z[i] : (int32_t)z[i];
            if (a > peak) { peak = a; }
        }
        s = (peak < ICHI_FEAT_PEAK_S0) ? 0U : ((peak < ICHI_FEAT_PEAK_S1) ? 1U : 2U);
        exponent = (uint8_t)(exponent + s);
        for (k = 0U; k < half; k++)
        {
            uint32_t tw = (uint32_t)k * (NC / half);
            int32_t c = cos_q(tw);
            int32_t sn = sin_q(tw);
            uint16_t a;
            for (a = k; a < NC; a = (uint16_t)(a + 2U * half))
            {
                uint16_t b = (uint16_t)(a + half);
                int32_t br = z[2U * b], bi = z[2U * b + 1U];
                int32_t ar = z[2U * a], ai = z[2U * a + 1U];
                int32_t tr = (br * c + bi * sn + 16384L) >> 15;
                int32_t ti = (bi * c - br * sn + 16384L) >> 15;
                z[2U * a] = (int16_t)shift_round(ar + tr, s);
                z[2U * a + 1U] = (int16_t)shift_round(ai + ti, s);
                z[2U * b] = (int16_t)shift_round(ar - tr, s);
                z[2U * b + 1U] = (int16_t)shift_round(ai - ti, s);
            }
        }
    }
    return exponent;
}

/* Windowed segment in z (int16 real samples, used as 1024 complex values)
   -> power of bins 1..1024 added to power_db. */
static void accumulate_segment(int16_t *z)
{
    uint16_t n, k;
    uint8_t exponent;
    float scale;

    for (n = 0U; n < ICHI_FEAT_NFFT; n++)
    {
        int32_t w = ichi_win_q15[(n < NC) ? n : (ICHI_FEAT_NFFT - 1U - n)];
        z[n] = (int16_t)(((int32_t)z[n] * w + 16384L) >> 15);
    }
    exponent = fft1024(z);
    if (exponent > last_exponent) { last_exponent = exponent; }
    scale = ldexpf(1.0f, 2 * (int)exponent - 30);   /* (2^exp / 32768)^2 */
    for (k = 1U; k <= NC; k++)
    {
        uint16_t kk = (uint16_t)(k % NC);
        uint16_t nk = (uint16_t)((NC - k) % NC);
        int64_t ar = z[2U * kk], ai = z[2U * kk + 1U];
        int64_t br = z[2U * nk], bi = -(int64_t)z[2U * nk + 1U];   /* conj(Z[N-k]) */
        int64_t er = ar + br, ei = ai + bi;                         /* 2E */
        int64_t orr = ai - bi, oi = -(ar - br);                     /* 2O = (A - B) / j */
        int64_t c = cos_q(k), sn = sin_q(k);
        int64_t wr = (orr * c + oi * sn + 16384) >> 15;
        int64_t wi = (oi * c - orr * sn + 16384) >> 15;
        float xr = (float)(er + wr);                                /* 2X */
        float xi = (float)(ei + wi);
        power_db[k - 1U] += ((xr * xr + xi * xi) * 0.25f) * scale;
    }
}

void IchiFeatureBaselineReset(void)
{
    uint16_t k;
    for (k = 0U; k < ICHI_FEAT_BINS; k++)
    {
        baseline_q[k] = 0;
    }
}

bool IchiFeatureFrame(IchiPcmReader reader, uint16_t total_samples)
{
    int16_t *z = (int16_t *)(void *)ichi_shared_scratch;
    uint16_t offset;
    uint16_t segments = 0U;
    uint16_t k;

    last_exponent = 0U;
    for (k = 0U; k < ICHI_FEAT_BINS; k++)
    {
        power_db[k] = 0.0f;
    }
    for (offset = 0U; (uint32_t)offset + ICHI_FEAT_NFFT <= total_samples;
         offset = (uint16_t)(offset + ICHI_FEAT_HOP))
    {
        if (!reader(offset, z, (uint16_t)ICHI_FEAT_NFFT))
        {
            return false;
        }
        accumulate_segment(z);
        segments++;
        wdt_clear();
    }
    if (segments == 0U)
    {
        return false;
    }
    for (k = 0U; k < ICHI_FEAT_BINS; k++)
    {
        float mean = power_db[k] / (float)segments;
        float db = 10.0f * log10f(mean + 1e-12f);
        power_db[k] = (db < -80.0f) ? -80.0f : db;
    }
    return true;
}

void IchiFeatureBaselineAdd(uint8_t frame_count)
{
    uint16_t k;
    for (k = 0U; k < ICHI_FEAT_BINS; k++)
    {
        baseline_q[k] = (int16_t)(baseline_q[k] +
                        round_half_even((power_db[k] * BASE_SCALE) / (float)frame_count));
    }
}

void IchiFeatureInput(int8_t out[ICHI_FEAT_INPUTS])
{
    double sum = 0.0;
    double var = 0.0;
    double mu64;
    float mu, sd;
    uint16_t k;

    for (k = 0U; k < ICHI_FEAT_BINS; k++)
    {
        power_db[k] = power_db[k] - (float)baseline_q[k] / BASE_SCALE;
        sum += (double)power_db[k];
    }
    mu64 = sum / (double)ICHI_FEAT_BINS;
    for (k = 0U; k < ICHI_FEAT_BINS; k++)
    {
        double d = (double)power_db[k] - mu64;
        var += d * d;
    }
    mu = (float)mu64;
    sd = (float)sqrt(var / (double)ICHI_FEAT_BINS);
    for (k = 0U; k < ICHI_FEAT_INPUTS; k++)
    {
        uint16_t bin = (uint16_t)(ICHI_FEAT_BIN_LO + k);
        float x = round_float16((power_db[bin] - mu) / (sd + 1e-6f));
        float xs = (x - ichi_feat_in_mu[k]) / ichi_feat_in_sd[k];
        int32_t v = round_half_even(xs / ICHI_FEAT_IN_SCALE);
        if (v > 127) { v = 127; }
        if (v < -127) { v = -127; }
        out[k] = (int8_t)v;
    }
}

uint8_t IchiFeatureLastExponent(void)
{
    return last_exponent;
}

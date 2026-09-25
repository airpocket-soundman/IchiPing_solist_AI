#include "ichi_tft.h"

#include <stddef.h>

#include "mcu.h"
#include "rdwr_reg.h"
#include "wdt.h"

/* Port mode field: 8 bits per pin (pins 0-3 in MODx0, 4-7 in MODx1); 0x02 = output. */
#define MODE_OUTPUT (0x02UL)
#define SCK_BIT     (1UL << 0)     /* P40 */
#define MOSI_BIT    (1UL << 1)     /* P41 */
#define PCA_OE_BIT  (1UL << 2)     /* P42: PCA9685 OE, keep High (servo PWM disabled) */
#define CS_BIT      (1UL << 3)     /* P43 */
#define DC_BIT      (1UL << 2)     /* P22 */
#define RST_BIT     (1UL << 3)     /* P23 */

/* Only touch the TFT bits: P45/P46 on the same port hold the board power keep and 5 V regulator. */
static void p4_set(uint32_t bits)   { PORT4->P4DO |= bits; }
static void p4_clear(uint32_t bits) { PORT4->P4DO &= ~bits; }
static void p2_set(uint32_t bits)   { PORT2->P2DO |= bits; }
static void p2_clear(uint32_t bits) { PORT2->P2DO &= ~bits; }

static void delay_ms(uint32_t ms)
{
    while (ms-- > 0U)
    {
        volatile uint32_t n = 48000UL / 6UL;   /* ~1 ms at 48 MHz (approximate) */
        while (n-- > 0U) { }
        wdt_clear();
    }
}

/* Two plain stores per bit: the other P4 outputs (P42 PCA OE, P45/P46 power) are
   captured once per byte so they keep their level. */
static void spi_byte(uint8_t value)
{
    volatile uint32_t *const out = &PORT4->P4DO;
    const uint32_t base = *out & ~(SCK_BIT | MOSI_BIT);
    uint8_t bit;
    for (bit = 0U; bit < 8U; bit++)
    {
        const uint32_t d = base | (((value & 0x80U) != 0U) ? MOSI_BIT : 0UL);
        *out = d;                                  /* data, SCK Low */
        *out = d | SCK_BIT;                        /* rising edge: sampled (mode 0) */
        value = (uint8_t)(value << 1);
    }
    *out = base;                                   /* SCK Low between bytes */
}

static void command(uint8_t cmd)
{
    p2_clear(DC_BIT);
    p4_clear(CS_BIT);
    spi_byte(cmd);
    p4_set(CS_BIT);
    p2_set(DC_BIT);
}

static void command_data(uint8_t cmd, const uint8_t *data, uint8_t n)
{
    uint8_t i;
    p2_clear(DC_BIT);
    p4_clear(CS_BIT);
    spi_byte(cmd);
    p2_set(DC_BIT);
    for (i = 0U; i < n; i++) { spi_byte(data[i]); }
    p4_set(CS_BIT);
}

static void set_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1)
{
    uint8_t col[4] = { (uint8_t)(x0 >> 8), (uint8_t)x0, (uint8_t)(x1 >> 8), (uint8_t)x1 };
    uint8_t row[4] = { (uint8_t)(y0 >> 8), (uint8_t)y0, (uint8_t)(y1 >> 8), (uint8_t)y1 };
    command_data(0x2AU, col, 4U);                  /* column address set */
    command_data(0x2BU, row, 4U);                  /* page address set */
    p2_clear(DC_BIT);
    p4_clear(CS_BIT);
    spi_byte(0x2CU);                               /* memory write; CS stays Low for pixel data */
    p2_set(DC_BIT);
}

void IchiTftInitialize(void)
{
    static const uint8_t colmod = 0x55U;           /* 16-bit RGB565 */
    static const uint8_t madctl = 0xE8U;           /* landscape (MY|MX|MV), BGR; 0x28 was upside down */

    /* Outputs: P40 SCK, P41 MOSI, P42 PCA OE (High), P43 CS (High), P22 DC, P23 RESET. */
    p4_set(CS_BIT | PCA_OE_BIT);
    p4_clear(SCK_BIT);
    p2_set(DC_BIT | RST_BIT);
    write_bit(PORT4->P4MOD0, 0xFFFFFFFFUL,
              (MODE_OUTPUT << 24) | (MODE_OUTPUT << 16) | (MODE_OUTPUT << 8) | MODE_OUTPUT);
    write_bit(PORT2->P2MOD0, 0xFFFF0000UL, (MODE_OUTPUT << 24) | (MODE_OUTPUT << 16));

    p2_clear(RST_BIT);                             /* hardware reset pulse */
    delay_ms(20U);
    p2_set(RST_BIT);
    delay_ms(150U);
    command(0x01U);                                /* software reset */
    delay_ms(150U);
    command(0x11U);                                /* sleep out */
    delay_ms(150U);
    command_data(0x3AU, &colmod, 1U);
    command_data(0x36U, &madctl, 1U);
    command(0x29U);                                /* display on */
    delay_ms(20U);
}

void IchiTftFillRect(uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint16_t color)
{
    uint32_t n = (uint32_t)w * h;
    if ((w == 0U) || (h == 0U)) { return; }
    set_window(x, y, (uint16_t)(x + w - 1U), (uint16_t)(y + h - 1U));
    while (n-- > 0U)
    {
        spi_byte((uint8_t)(color >> 8));
        spi_byte((uint8_t)color);
        if ((n & 0x3FFFUL) == 0U) { wdt_clear(); }
    }
    p4_set(CS_BIT);
}

/* Classic 5x7 font, columns LSB = top row, for ' ' (0x20) .. 'Z' (0x5A). */
static const uint8_t font5x7[59][5] = {
    {0x00,0x00,0x00,0x00,0x00},{0x00,0x00,0x5F,0x00,0x00},{0x00,0x07,0x00,0x07,0x00},{0x14,0x7F,0x14,0x7F,0x14},
    {0x24,0x2A,0x7F,0x2A,0x12},{0x23,0x13,0x08,0x64,0x62},{0x36,0x49,0x55,0x22,0x50},{0x00,0x05,0x03,0x00,0x00},
    {0x00,0x1C,0x22,0x41,0x00},{0x00,0x41,0x22,0x1C,0x00},{0x14,0x08,0x3E,0x08,0x14},{0x08,0x08,0x3E,0x08,0x08},
    {0x00,0x50,0x30,0x00,0x00},{0x08,0x08,0x08,0x08,0x08},{0x00,0x60,0x60,0x00,0x00},{0x20,0x10,0x08,0x04,0x02},
    {0x3E,0x51,0x49,0x45,0x3E},{0x00,0x42,0x7F,0x40,0x00},{0x42,0x61,0x51,0x49,0x46},{0x21,0x41,0x45,0x4B,0x31},
    {0x18,0x14,0x12,0x7F,0x10},{0x27,0x45,0x45,0x45,0x39},{0x3C,0x4A,0x49,0x49,0x30},{0x01,0x71,0x09,0x05,0x03},
    {0x36,0x49,0x49,0x49,0x36},{0x06,0x49,0x49,0x29,0x1E},{0x00,0x36,0x36,0x00,0x00},{0x00,0x56,0x36,0x00,0x00},
    {0x08,0x14,0x22,0x41,0x00},{0x14,0x14,0x14,0x14,0x14},{0x00,0x41,0x22,0x14,0x08},{0x02,0x01,0x51,0x09,0x06},
    {0x32,0x49,0x79,0x41,0x3E},{0x7E,0x11,0x11,0x11,0x7E},{0x7F,0x49,0x49,0x49,0x36},{0x3E,0x41,0x41,0x41,0x22},
    {0x7F,0x41,0x41,0x22,0x1C},{0x7F,0x49,0x49,0x49,0x41},{0x7F,0x09,0x09,0x09,0x01},{0x3E,0x41,0x49,0x49,0x7A},
    {0x7F,0x08,0x08,0x08,0x7F},{0x00,0x41,0x7F,0x41,0x00},{0x20,0x40,0x41,0x3F,0x01},{0x7F,0x08,0x14,0x22,0x41},
    {0x7F,0x40,0x40,0x40,0x40},{0x7F,0x02,0x0C,0x02,0x7F},{0x7F,0x04,0x08,0x10,0x7F},{0x3E,0x41,0x41,0x41,0x3E},
    {0x7F,0x09,0x09,0x09,0x06},{0x3E,0x41,0x51,0x21,0x5E},{0x7F,0x09,0x19,0x29,0x46},{0x46,0x49,0x49,0x49,0x31},
    {0x01,0x01,0x7F,0x01,0x01},{0x3F,0x40,0x40,0x40,0x3F},{0x1F,0x20,0x40,0x20,0x1F},{0x3F,0x40,0x38,0x40,0x3F},
    {0x63,0x14,0x08,0x14,0x63},{0x07,0x08,0x70,0x08,0x07},{0x61,0x51,0x49,0x45,0x43},
};

void IchiTftDrawText(uint16_t x, uint16_t y, const char *text, uint16_t fg, uint16_t bg, uint8_t scale)
{
    for (; (text != NULL) && (*text != '\0'); text++, x = (uint16_t)(x + 6U * scale))
    {
        char c = *text;
        const uint8_t *glyph;
        uint8_t col, row, sy, sx;
        if ((c >= 'a') && (c <= 'z')) { c = (char)(c - ('a' - 'A')); }
        if ((c < ' ') || (c > 'Z')) { c = ' '; }
        glyph = font5x7[(uint8_t)(c - ' ')];
        set_window(x, y, (uint16_t)(x + 6U * scale - 1U), (uint16_t)(y + 8U * scale - 1U));
        for (row = 0U; row < 8U; row++)
        {
            for (sy = 0U; sy < scale; sy++)
            {
                for (col = 0U; col < 6U; col++)
                {
                    uint16_t color = ((col < 5U) && (row < 7U) && ((glyph[col] >> row) & 1U)) ? fg : bg;
                    for (sx = 0U; sx < scale; sx++)
                    {
                        spi_byte((uint8_t)(color >> 8));
                        spi_byte((uint8_t)color);
                    }
                }
            }
        }
        p4_set(CS_BIT);
        wdt_clear();
    }
}

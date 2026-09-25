#ifndef ICHI_TFT_H
#define ICHI_TFT_H

/* ILI9341 2.4" 240x320 SPI TFT on the Stamp-S3A interposer (CN3):
     P40 = SCK, P41 = MOSI, P43 = CS (10 k pull-up), P22 = D/C, P23 = RESET.
   Write-only; bit-banged SPI (mode 0) on GPIO for bring-up.  Landscape 320x240, RGB565. */

#include <stdint.h>

#define ICHI_TFT_WIDTH  (320U)
#define ICHI_TFT_HEIGHT (240U)

#define ICHI_RGB565(r, g, b) ((uint16_t)((((r) & 0xF8U) << 8) | (((g) & 0xFCU) << 3) | ((b) >> 3)))
#define ICHI_BLACK   ICHI_RGB565(0, 0, 0)
#define ICHI_WHITE   ICHI_RGB565(255, 255, 255)
#define ICHI_RED     ICHI_RGB565(255, 0, 0)
#define ICHI_GREEN   ICHI_RGB565(0, 255, 0)
#define ICHI_BLUE    ICHI_RGB565(0, 0, 255)
#define ICHI_YELLOW  ICHI_RGB565(255, 255, 0)
#define ICHI_CYAN    ICHI_RGB565(0, 255, 255)
#define ICHI_MAGENTA ICHI_RGB565(255, 0, 255)

void IchiTftInitialize(void);
void IchiTftFillRect(uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint16_t color);
/* 5x7 font (space, digits, A-Z, a few symbols), each dot drawn as scale x scale pixels. */
void IchiTftDrawText(uint16_t x, uint16_t y, const char *text, uint16_t fg, uint16_t bg, uint8_t scale);

#endif

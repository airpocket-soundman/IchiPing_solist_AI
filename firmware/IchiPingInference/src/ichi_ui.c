#include "ichi_ui.h"

#include <string.h>

#include "ichi_tft.h"

#define NAVY        (0x000FU)
#define GREY        (0x8410U)
#define ORANGE      (0xFD20U)
#define DARK_GREEN  (0x03E0U)
#define DARK_RED    (0x7800U)
#define DARK_ORANGE (0x7A80U)
#define GREEN       (0x07E0U)
#define RED         (0xF800U)
#define BLUE        (0x001FU)

#define DIGIT_SCALE (5U)
#define DIGIT_X     (85U)
#define DIGIT_STEP  (30U)
#define ROW_INF_Y   (40U)
#define ROW_ACT_Y   (100U)
#define INFO_Y      (150U)
#define BANNER_X    (4U)
#define BANNER_Y    (175U)
#define BANNER_W    (312U)
#define BANNER_H    (46U)

/* Screen slot -> state bit: c, BC, b, AB, a. */
static const uint8_t slot_bit[5] = { 2U, 4U, 1U, 3U, 0U };

static char shown_char[2][5];
static uint16_t shown_color[2][5];
static char shown_banner[24];
static uint16_t shown_banner_bg;

/* a and AB are always audible; b and BC need AB open; c needs AB and BC open. */
static bool observable(uint8_t bit, uint8_t actual)
{
    bool ab = ((actual >> 3) & 1U) != 0U;
    bool bc = ((actual >> 4) & 1U) != 0U;
    if ((bit == 0U) || (bit == 3U)) { return true; }
    if ((bit == 1U) || (bit == 4U)) { return ab; }
    return ab && bc;
}

static uint8_t class14(uint8_t s)
{
    uint8_t a = s & 1U, b = (s >> 1) & 1U, c = (s >> 2) & 1U, ab = (s >> 3) & 1U, bc = (s >> 4) & 1U;
    if (ab == 0U) { return a; }
    if (bc == 0U) { return (uint8_t)(2U + a + 2U * b); }
    return (uint8_t)(6U + a + 2U * b + 4U * c);
}

static void draw_digit(uint8_t row, uint8_t slot, char ch, uint16_t color)
{
    char text[2] = { ch, '\0' };
    if ((shown_char[row][slot] == ch) && (shown_color[row][slot] == color)) { return; }
    IchiTftDrawText((uint16_t)(DIGIT_X + slot * DIGIT_STEP), (row == 0U) ? ROW_INF_Y : ROW_ACT_Y,
                    text, color, ICHI_BLACK, DIGIT_SCALE);
    shown_char[row][slot] = ch;
    shown_color[row][slot] = color;
}

void IchiUiBanner(const char *text, uint16_t background, uint16_t foreground)
{
    uint16_t len = (uint16_t)strlen(text);
    if ((background == shown_banner_bg) && (strcmp(text, shown_banner) == 0)) { return; }
    IchiTftFillRect(BANNER_X, BANNER_Y, BANNER_W, BANNER_H, background);
    if (len > 0U)
    {
        uint16_t w = (uint16_t)(len * 12U);
        IchiTftDrawText((uint16_t)(BANNER_X + ((w < BANNER_W) ? (BANNER_W - w) / 2U : 0U)), BANNER_Y + 16U,
                        text, foreground, background, 2U);
    }
    strncpy(shown_banner, text, sizeof(shown_banner) - 1U);
    shown_banner_bg = background;
}

void IchiUiInitialize(void)
{
    IchiTftFillRect(0U, 0U, ICHI_TFT_WIDTH, ICHI_TFT_HEIGHT, ICHI_BLACK);
    IchiTftFillRect(0U, 0U, ICHI_TFT_WIDTH, 28U, NAVY);
    IchiTftDrawText(6U, 7U, "IchiPing infer", ICHI_WHITE, NAVY, 2U);
    IchiTftDrawText(6U, 50U, "inf", ICHI_WHITE, ICHI_BLACK, 2U);
    IchiTftDrawText(6U, 110U, "act", ICHI_WHITE, ICHI_BLACK, 2U);
    memset(shown_char, 0, sizeof(shown_char));
    memset(shown_color, 0, sizeof(shown_color));
    shown_banner[0] = '\0';
    shown_banner_bg = ICHI_BLACK;
    IchiUiShowState(ICHI_UI_NONE, ICHI_UI_NONE);
}

void IchiUiShowState(uint8_t actual, uint8_t inferred)
{
    uint8_t slot;
    for (slot = 0U; slot < 5U; slot++)
    {
        uint8_t bit = slot_bit[slot];
        bool obs = (actual != ICHI_UI_NONE) ? observable(bit, actual) : true;
        char act_ch = (actual != ICHI_UI_NONE) ? (char)('0' + ((actual >> bit) & 1U)) : '-';
        char inf_ch = (inferred != ICHI_UI_NONE) ? (char)('0' + ((inferred >> bit) & 1U)) : '-';
        uint16_t inf_color = GREY;
        if ((inferred != ICHI_UI_NONE) && (actual != ICHI_UI_NONE))
        {
            bool ok = (((inferred ^ actual) >> bit) & 1U) == 0U;
            inf_color = ok ? (obs ? GREEN : DARK_GREEN) : (obs ? RED : DARK_RED);
        }
        draw_digit(0U, slot, inf_ch, inf_color);
        draw_digit(1U, slot, act_ch, (actual == ICHI_UI_NONE) ? GREY : (obs ? ORANGE : DARK_ORANGE));
    }
    if ((inferred != ICHI_UI_NONE) && (actual != ICHI_UI_NONE))
    {
        if (inferred == actual) { IchiUiBanner("Complete Success", BLUE, ICHI_WHITE); }
        else if (class14(inferred) == class14(actual)) { IchiUiBanner("Conditional Success", GREEN, ICHI_WHITE); }
        else { IchiUiBanner("Failure", RED, ICHI_WHITE); }
    }
}

void IchiUiStateLabel(char *dst, uint8_t state)
{
    uint8_t slot;
    dst[0] = 'h';
    for (slot = 0U; slot < 5U; slot++)
    {
        dst[1U + slot] = (char)('0' + ((state >> slot_bit[slot]) & 1U));
    }
}

void IchiUiInfoLine(const char *text, uint16_t color)
{
    IchiTftFillRect(0U, INFO_Y, ICHI_TFT_WIDTH, 16U, ICHI_BLACK);
    IchiTftDrawText(6U, INFO_Y, text, color, ICHI_BLACK, 2U);
}

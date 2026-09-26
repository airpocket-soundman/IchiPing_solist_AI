#ifndef ICHI_UI_H
#define ICHI_UI_H

/* TFT inference screen, same layout as the original IchiPing 10_inference
   (D:/GitHub/IchiPing firmware/projects/10_inference/main.c; UNO Q ported it as-is):
     header  (0,0,320,28) navy, "IchiPing infer"
     "inf" row y=40 / "act" row y=100: five 0/1 digits at scale 5, x = 85 + 30*slot,
       slot order c, BC, b, AB, a (physical layout)
     inf digits: GREEN correct / RED wrong; unobservable openings (behind closed doors)
       DARK_GREEN / DARK_RED; GREY "-----" without a result
     act digits: ORANGE observable / DARK_ORANGE unobservable
     banner (4,175,312,46): BLUE "Complete Success" (32-class), GREEN "Conditional Success"
       (14-class equivalent), RED "Failure"; status messages in between.
   Only the digits / banner that changed are redrawn. */

#include <stdbool.h>
#include <stdint.h>

#define ICHI_UI_NONE (0xFFU)          /* no state / no result */

void IchiUiInitialize(void);
/* actual: door/window bits (a=bit0 .. BC=bit4) or ICHI_UI_NONE; inferred: prediction or
   ICHI_UI_NONE.  With both present the banner shows the verdict. */
void IchiUiShowState(uint8_t actual, uint8_t inferred);
/* Status message in the banner (cleared by the next verdict or message). */
void IchiUiBanner(const char *text, uint16_t background, uint16_t foreground);
/* "h" + five 0/1 in screen order C, BC, B, AB, A (1 = OPEN), e.g. "h01101"; writes 6 chars. */
void IchiUiStateLabel(char *dst, uint8_t state);
/* One line of small text between the rows and the banner (survey progress). */
void IchiUiInfoLine(const char *text, uint16_t color);

#endif

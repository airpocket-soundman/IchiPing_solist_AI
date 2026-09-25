#ifndef ICHI_APP_H
#define ICHI_APP_H

#include <stdbool.h>

void IchiAppInitialize(bool lcd_available);
/* Call from the main loop: handles one pending UART request at a time. */
void IchiAppProcess(void);

#endif

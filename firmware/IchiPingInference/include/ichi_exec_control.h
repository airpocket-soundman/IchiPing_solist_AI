#ifndef ICHI_EXEC_CONTROL_H
#define ICHI_EXEC_CONTROL_H

#include <stdbool.h>
#include <stdint.h>

/* 10 ms ticks. Both destructive reset and ODL require a separate EXEC press. */
typedef enum {
    ICHI_EXEC_NONE, ICHI_EXEC_INFER, ICHI_EXEC_SHOW_ADD,
    ICHI_EXEC_CONFIRM_ADD, ICHI_EXEC_CONFIRM_RESET,
    ICHI_EXEC_START_ADD, ICHI_EXEC_RESET_ODL, ICHI_EXEC_CANCEL
} IchiExecAction;

typedef struct {
    uint32_t start;
    uint8_t pending; /* 0 none, 1 add ODL, 2 reset ODL */
    bool pressed;
    bool announced;
    bool ignore_until_release;
} IchiExecControl;

static inline void IchiExecControlInit(IchiExecControl *s, bool down)
{
    s->start = 0U; s->pending = 0U; s->pressed = down;
    s->announced = false; s->ignore_until_release = down;
}

static inline void IchiExecRequestAdd(IchiExecControl *s, bool down)
{
    s->pending = 1U; s->pressed = down; s->ignore_until_release = false;
}

static inline IchiExecAction IchiExecControlUpdate(IchiExecControl *s,
                                                  uint32_t now, bool down, bool all_open)
{
    uint32_t held;
    if (s->ignore_until_release) {
        s->pressed = down;
        if (!down) { s->ignore_until_release = false; }
        return ICHI_EXEC_NONE;
    }
    if (s->pending != 0U) {
        if (down && !s->pressed) {
            IchiExecAction action = !all_open ? ICHI_EXEC_CANCEL :
                ((s->pending == 1U) ? ICHI_EXEC_START_ADD : ICHI_EXEC_RESET_ODL);
            s->pending = 0U; s->pressed = true; s->ignore_until_release = true;
            return action;
        }
        s->pressed = down;
        return ICHI_EXEC_NONE;
    }
    if (down) {
        if (!s->pressed) { s->start = now; s->announced = false; s->pressed = true; }
        held = now - s->start;
        if (held >= 1000U) { s->pending = 2U; return ICHI_EXEC_CONFIRM_RESET; }
        if ((held >= 200U) && !s->announced) { s->announced = true; return ICHI_EXEC_SHOW_ADD; }
    } else if (s->pressed) {
        held = now - s->start;
        s->pressed = false;
        if (held >= 1000U) { s->pending = 2U; return ICHI_EXEC_CONFIRM_RESET; }
        if (held >= 200U) { s->pending = 1U; return ICHI_EXEC_CONFIRM_ADD; }
        return ICHI_EXEC_INFER;
    }
    return ICHI_EXEC_NONE;
}

#endif

/* Freestanding test: returns the failing source line, or zero on success. */
#include "ichi_exec_control.h"

#define CHECK(value) do { if (!(value)) { return __LINE__; } } while (0)
#define UPDATE(t, down, yes) IchiExecControlUpdate(&s, (t), (down), (yes))

uint32_t run_tests(void)
{
    IchiExecControl s;
    uint32_t t;
    IchiExecControlInit(&s, false);
    CHECK(UPDATE(100U, true, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(299U, false, true) == ICHI_EXEC_INFER);

    /* Prompt once at 2 s. Neither holding nor releasing starts ODL. */
    CHECK(UPDATE(400U, true, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(599U, true, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(600U, true, true) == ICHI_EXEC_SHOW_ADD);
    for (t = 601U; t < 1400U; t++) { CHECK(UPDATE(t, true, true) == ICHI_EXEC_NONE); }
    CHECK(UPDATE(1399U, false, true) == ICHI_EXEC_CONFIRM_ADD);
    CHECK(UPDATE(1400U, false, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(1401U, true, true) == ICHI_EXEC_START_ADD);
    CHECK(UPDATE(2500U, true, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(2501U, false, true) == ICHI_EXEC_NONE);

    /* Any non-all-open selection cancels, even if EXEC remains held. */
    CHECK(UPDATE(3000U, true, false) == ICHI_EXEC_NONE);
    CHECK(UPDATE(3200U, false, false) == ICHI_EXEC_CONFIRM_ADD);
    CHECK(UPDATE(3201U, true, false) == ICHI_EXEC_CANCEL);
    CHECK(UPDATE(4300U, true, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(4301U, false, true) == ICHI_EXEC_NONE);

    /* At 10 s ask for reset; reset only on a fresh confirmed EXEC press. */
    CHECK(UPDATE(5000U, true, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(5200U, true, true) == ICHI_EXEC_SHOW_ADD);
    CHECK(UPDATE(5500U, true, true) == ICHI_EXEC_NONE); /* no 5-second mode */
    CHECK(UPDATE(6000U, true, true) == ICHI_EXEC_CONFIRM_RESET);
    CHECK(UPDATE(6200U, true, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(6201U, false, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(6202U, true, true) == ICHI_EXEC_RESET_ODL);
    CHECK(UPDATE(6203U, false, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(6300U, true, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(6500U, false, true) == ICHI_EXEC_CONFIRM_ADD);
    CHECK(UPDATE(6501U, true, true) == ICHI_EXEC_START_ADD);
    CHECK(UPDATE(6502U, false, true) == ICHI_EXEC_NONE);

    CHECK(UPDATE(7000U, true, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(8000U, false, false) == ICHI_EXEC_CONFIRM_RESET);
    CHECK(UPDATE(8001U, true, false) == ICHI_EXEC_CANCEL);
    CHECK(UPDATE(8002U, false, false) == ICHI_EXEC_NONE);

    /* A button held at power-on must be released before any gesture. */
    IchiExecControlInit(&s, true);
    CHECK(UPDATE(2000U, true, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(2001U, false, true) == ICHI_EXEC_NONE);

    /* Unsigned timer wrap and sparse polling at the threshold. */
    CHECK(UPDATE(0xFFFFFF80U, true, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(0x00000048U, false, true) == ICHI_EXEC_CONFIRM_ADD);
    CHECK(UPDATE(0x00000049U, true, false) == ICHI_EXEC_CANCEL);
    CHECK(UPDATE(0x00000050U, false, false) == ICHI_EXEC_NONE);

    /* Remote ODL requests also use the same physical confirmation. */
    IchiExecRequestAdd(&s, true);
    CHECK(UPDATE(100U, true, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(101U, false, true) == ICHI_EXEC_NONE);
    CHECK(UPDATE(102U, true, true) == ICHI_EXEC_START_ADD);
    return 0U;
}

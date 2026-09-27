################################################################################
# IchiPing sources (tools/build.ps1 installs this as Debug/S_IchiPing/subdir.mk)
################################################################################
C_SRCS += ../S_IchiPing/ichi_protocol.c ../S_IchiPing/ichi_inference.c ../S_IchiPing/ichi_feature.c
RESS += ./S_IchiPing/ichi_protocol.res ./S_IchiPing/ichi_inference.res ./S_IchiPing/ichi_feature.res
RESS__QUOTED += "./S_IchiPing/ichi_protocol.res" "./S_IchiPing/ichi_inference.res" "./S_IchiPing/ichi_feature.res"
ASMS += ./S_IchiPing/ichi_protocol.asm ./S_IchiPing/ichi_inference.asm ./S_IchiPing/ichi_feature.asm
ASMS__QUOTED += "./S_IchiPing/ichi_protocol.asm" "./S_IchiPing/ichi_inference.asm" "./S_IchiPing/ichi_feature.asm"
OBJS += ./S_IchiPing/ichi_protocol.o ./S_IchiPing/ichi_inference.o ./S_IchiPing/ichi_feature.o
OBJS__QUOTED += "./S_IchiPing/ichi_protocol.o" "./S_IchiPing/ichi_inference.o" "./S_IchiPing/ichi_feature.o"

S_IchiPing/%.asm: ../S_IchiPing/%.c
	lccarm @"./S_IchiPing/$*.res"

S_IchiPing/%.res: S_IchiPing/%.asm
S_IchiPing/%.i: S_IchiPing/%.asm

S_IchiPing/%.o: ./S_IchiPing/%.asm
	llvm-mc-arm -g -dwarf-version=4 -filetype=obj -o="$@" -mcpu=cortex-m0plus "$<"

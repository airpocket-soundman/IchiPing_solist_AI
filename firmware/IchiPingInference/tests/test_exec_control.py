"""Compile and execute the real button controller on an emulated Cortex-M0+.

Requires the installed LEXIDE toolchain and `python -m pip install unicorn`.
No board is connected or modified by this test.
"""
from pathlib import Path
import subprocess

from unicorn import Uc, UC_ARCH_ARM, UC_MODE_THUMB
from unicorn.arm_const import UC_ARM_REG_LR, UC_ARM_REG_R0, UC_ARM_REG_SP

root = Path(__file__).resolve().parents[3]
firmware = root / 'firmware/IchiPingInference'
toolchain = sorted(Path('C:/LAPIS/LEXIDE/BuildTools').glob('Ver.*'))[-1]
out = root / '.local/exec-control-test'
out.mkdir(parents=True, exist_ok=True)

def run(tool, *args):
    subprocess.run([str(toolchain/'Bin'/tool), *map(str, args)], check=True)

run('clang-arm.exe', '-target', 'arm-none-eabi', '-mcpu=cortex-m0plus', '-mthumb',
    '-ffreestanding', '-fno-builtin', '-Os', '-I', toolchain/'Inc/Arm/Clang',
    '-I', firmware/'include', '-c', firmware/'tests/exec_control_test.c', '-o', out/'test.o')
run('lld-arm.exe', '-Ttext=0x10000', '--entry=run_tests', out/'test.o', '-o', out/'test.elf')
run('llvm-objcopy.exe', '-O', 'binary', out/'test.elf', out/'test.bin')

cpu = Uc(UC_ARCH_ARM, UC_MODE_THUMB)
cpu.mem_map(0x10000, 0x20000)
cpu.mem_map(0x20000000, 0x10000)
cpu.mem_write(0x10000, (out/'test.bin').read_bytes())
cpu.reg_write(UC_ARM_REG_SP, 0x2000FFF0)
cpu.reg_write(UC_ARM_REG_LR, 0x20001)
cpu.emu_start(0x10001, 0x20000, count=1000000)
failure = cpu.reg_read(UC_ARM_REG_R0)
if failure:
    raise AssertionError(f'exec_control_test.c:{failure}')
print('PASS: release timing, separate confirmation, cancellation, reset, boot-held button and timer wrap')

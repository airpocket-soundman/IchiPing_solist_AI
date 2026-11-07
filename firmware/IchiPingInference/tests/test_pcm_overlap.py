"""Execute the production PCM window loop on Cortex-M0+ with a destructive FFT stub.

Checks exact sample order, no duplicate reads, frame restart, and failed transfers.
Requires LEXIDE and Unicorn; does not access the board.
"""
from pathlib import Path
import subprocess
import struct
from unicorn import Uc, UC_ARCH_ARM, UC_MODE_THUMB
from unicorn.arm_const import UC_ARM_REG_LR, UC_ARM_REG_R0, UC_ARM_REG_SP

root = Path(__file__).resolve().parents[3]
fw = root / 'firmware/IchiPingInference'
out = root / '.local/pcm-overlap-test'
out.mkdir(parents=True, exist_ok=True)
source = (fw/'src/ichi_feature.c').read_text()
loop = source.split('    for (offset = 0U;', 1)[1].split('    if (segments == 0U)', 1)[0]
loop = '    for (offset = 0U;' + loop
test = r'''
#include <stdint.h>
#include <stdbool.h>
#define ICHI_FEAT_NFFT 2048U
#define ICHI_FEAT_HOP 1024U
#define ICHI_PCM_OVERLAP_CACHE 1
static int16_t pcm_overlap[1024], z[2048];
static uint32_t read_end, read_count, windows, bad, fail_at, salt;
static int16_t sample(uint32_t i) { return (int16_t)(i * 197U + salt); }
static bool reader(uint16_t offset, int16_t *dst, uint16_t count) {
    uint16_t i;
    if (offset != read_end) bad = 1;
    if (offset >= fail_at) return false;
    read_end += count; read_count += count;
    for (i=0; i<count; i++) dst[i] = sample(offset+i);
    return true;
}
static void accumulate_segment(int16_t *p) {
    uint16_t i;
    for(i=0; i<2048; i++) {
        if (p[i] != sample(windows * 1024U + i)) bad = 2;
        p[i] = (int16_t)0xDEAD; /* FFT destroys every raw sample. */
    }
    windows++;
}
static void wdt_clear(void) {}
static bool frame(uint16_t total_samples) {
    uint16_t offset, segments=0, k;
''' + loop + r'''
    return segments != 0;
}
static void reset(uint32_t s, uint32_t fail) {
    salt=s; fail_at=fail; read_end=read_count=windows=bad=0;
}
__attribute__((section(".text.run_tests"))) int run_tests(void) {
    reset(3,65536);
    if (!frame(32000) || bad || windows!=30 || read_count!=31744) return 1;
    reset(123,65536);
    if (!frame(32000) || bad || windows!=30 || read_count!=31744) return 2;
    reset(7,2048);
    if (frame(32000) || bad || windows!=1) return 3;
    reset(42,65536);
    if (!frame(32000) || bad || read_count!=31744) return 4;
    reset(11,65536);
    if (!frame(2048) || bad || windows!=1 || read_count!=2048) return 5;
    reset(11,65536);
    if (frame(2047) || read_count) return 6;
    return 0;
}
'''
(out/'test.c').write_text(test)
tc = sorted(Path('C:/LAPIS/LEXIDE/BuildTools').glob('Ver.*'))[-1]
def run(tool, *args):
    subprocess.run([str(tc/'Bin'/tool), *map(str,args)], check=True)
run('clang-arm.exe','-target','arm-none-eabi','-mcpu=cortex-m0plus','-mthumb',
    '-ffreestanding','-fno-builtin','-Os','-I',tc/'Inc/Arm/Clang',
    '-c',out/'test.c','-o',out/'test.o')
run('lld-arm.exe','-Ttext=0x10000','-Tbss=0x20000000','--entry=run_tests',out/'test.o','-o',out/'test.elf')
cpu=Uc(UC_ARCH_ARM,UC_MODE_THUMB)
cpu.mem_map(0x10000,0x20000)
cpu.mem_map(0x20000000,0x20000)
elf=(out/'test.elf').read_bytes()
header=struct.unpack_from('<16sHHIIIIIHHHHHH',elf)
entry=header[4]
for i in range(header[10]):
    kind,offset,address,_,size,_,_,_=struct.unpack_from('<IIIIIIII',elf,header[5]+i*header[9])
    if kind==1 and size:
        cpu.mem_write(address,elf[offset:offset+size])
cpu.reg_write(UC_ARM_REG_SP,0x2000FFF0)
cpu.reg_write(UC_ARM_REG_LR,0x20001)
cpu.emu_start(entry | 1,0x20000,count=20000000)
result=cpu.reg_read(UC_ARM_REG_R0)
assert result==0, f'PCM overlap check {result} failed'
print('PASS: exact raw windows after destructive FFT, 31744 unique reads, restart and failure recovery')

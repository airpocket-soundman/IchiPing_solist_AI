"""Render manual PNGs by running the real ichi_ui.c on an emulated Cortex-M0+.

Requires Pillow, unicorn and the installed LEXIDE ARM tools. No device access.
Example: python tools/render_manual_ui.py
Build artifacts and optional local Python packages live in ignored .local/.
"""
from pathlib import Path
import hashlib
import json
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.local/manual-python'))
from PIL import Image, ImageDraw
from unicorn import Uc, UC_ARCH_ARM, UC_MODE_THUMB, UC_HOOK_CODE
from unicorn.arm_const import UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3, UC_ARM_REG_SP, UC_ARM_REG_LR, UC_ARM_REG_PC

FW = ROOT / 'firmware/IchiPingInference'
OUT = ROOT / 'docs/manual/assets/screens'
BUILD = ROOT / '.local/manual-ui'
OUT.mkdir(parents=True, exist_ok=True)
BUILD.mkdir(parents=True, exist_ok=True)
TC = sorted(Path('C:/LAPIS/LEXIDE/BuildTools').glob('Ver.*'))[-1]

# Drawing calls are intercepted on the host. All layout, colors, observable-bit
# rules, success classification and progress painting run in the real UI C code.
SHIM = r'''
#include "ichi_ui.h"
#include "ichi_tft.h"
#include <stddef.h>
unsigned __aeabi_uidiv(unsigned a,unsigned b) { unsigned q=0,r=0; for(int i=31;i>=0;i--){r=(r<<1)|((a>>i)&1U);if(r>=b){r-=b;q|=1U<<i;}} return q; }
void *memset(void *p, int c, size_t n) { unsigned char *q=p; while(n--) *q++=c; return p; }
size_t strlen(const char *p) { size_t n=0; while(p[n]) n++; return n; }
int strcmp(const char *a,const char *b) { while(*a && *a==*b){a++;b++;} return (unsigned char)*a-(unsigned char)*b; }
char *strncpy(char *d,const char *s,size_t n) {char *r=d; while(n && *s){*d++=*s++;n--;} while(n--)*d++=0;return r;}
__attribute__((noinline)) void IchiTftFillRect(uint16_t x,uint16_t y,uint16_t w,uint16_t h,uint16_t c) { __asm volatile("nop"); }
__attribute__((noinline)) void IchiTftDrawText(uint16_t x,uint16_t y,const char *s,uint16_t f,uint16_t b,uint8_t z) { __asm volatile("nop"); }
void render(unsigned scene) {
 IchiUiInitialize();
 IchiUiShowState(31,ICHI_UI_NONE);
 if(scene==0) { IchiUiShowState(0,ICHI_UI_NONE); IchiUiBanner("Closing all",ICHI_YELLOW,ICHI_BLACK); }
 if(scene==1) { IchiUiShowState(0,ICHI_UI_NONE); IchiUiInfoLine("baseline 2/3",ICHI_CYAN); IchiUiBanner("Calibrating...",ICHI_CYAN,ICHI_BLACK); }
 if(scene==2) { IchiUiShowState(0,ICHI_UI_NONE); IchiUiInfoLine("calibrated",ICHI_WHITE); IchiUiBanner("Baseline ready",ICHI_GREEN,ICHI_WHITE); }
 if(scene==3) { IchiUiInfoLine("calibrated",ICHI_WHITE); IchiUiBanner("Listening...",ICHI_CYAN,ICHI_BLACK); }
 if(scene==4) { IchiUiInfoLine("calibrated",ICHI_WHITE); IchiUiProgress("Transferring...",0,ICHI_RGB565(0,150,255),ICHI_RGB565(0,0,72),ICHI_WHITE); IchiUiProgress("Transferring...",550,ICHI_RGB565(0,150,255),ICHI_RGB565(0,0,72),ICHI_WHITE); }
 if(scene==5) { IchiUiInfoLine("calibrated",ICHI_WHITE); IchiUiProgress("Inferring...",0,ICHI_RGB565(0,150,255),ICHI_RGB565(0,0,72),ICHI_WHITE); IchiUiProgress("Inferring...",950,ICHI_RGB565(0,150,255),ICHI_RGB565(0,0,72),ICHI_WHITE); }
 if(scene==6) { IchiUiInfoLine("calibrated",ICHI_WHITE); IchiUiShowState(31,31); }
 if(scene==7) { IchiUiInfoLine("calibrated",ICHI_WHITE); IchiUiShowState(1,5); }
 if(scene==8) { IchiUiInfoLine("calibrated",ICHI_WHITE); IchiUiShowState(31,30); }
 if(scene==9) { IchiUiBanner("Release: add ODL",ICHI_MAGENTA,ICHI_WHITE); IchiUiInfoLine("Hold 10s: reset ODL",ICHI_WHITE); }
 if(scene==10) { IchiUiBanner("Add ODL?",ICHI_MAGENTA,ICHI_WHITE); IchiUiInfoLine("All OPEN + EXEC: yes",ICHI_WHITE); }
 if(scene==11) { IchiUiShowState(10,ICHI_UI_NONE); IchiUiBanner("Additional ODL",ICHI_MAGENTA,ICHI_WHITE); IchiUiInfoLine("cal 13/32 h00010",ICHI_MAGENTA); }
 if(scene==12) { IchiUiShowState(16,ICHI_UI_NONE); IchiUiBanner("Calibrated",ICHI_GREEN,ICHI_WHITE); IchiUiInfoLine("post-update 133/160",ICHI_WHITE); }
 if(scene==13) { IchiUiBanner("Reset ODL?",ICHI_MAGENTA,ICHI_WHITE); IchiUiInfoLine("All OPEN + EXEC: yes",ICHI_WHITE); }
 if(scene==14) { IchiUiBanner("ODL reset",ICHI_GREEN,ICHI_WHITE); IchiUiInfoLine("Pretrained model",ICHI_WHITE); }
 if(scene==15) { IchiUiBanner("Canceled",ICHI_YELLOW,ICHI_BLACK); IchiUiInfoLine("calibrated",ICHI_WHITE); }
 if(scene==16) { IchiUiShowState(10,ICHI_UI_NONE); IchiUiBanner("Calibration aborted",ICHI_RED,ICHI_WHITE); IchiUiInfoLine("Previous ODL restored",ICHI_WHITE); }
 if(scene==17) { IchiUiBanner("Error stage 4",ICHI_RED,ICHI_WHITE); }
 if(scene==18) { IchiUiShowState(0,ICHI_UI_NONE); IchiUiBanner("New ODL",ICHI_MAGENTA,ICHI_WHITE); IchiUiInfoLine("cal 01/32 h00000",ICHI_MAGENTA); }
}
'''
# Gray-code step 13 is state 10, whose screen order is h00110.
SHIM = SHIM.replace('cal 13/32 h00010', 'cal 13/32 h00110')
(BUILD / 'shim.c').write_text(SHIM, encoding='utf-8')
def run(tool, *args):
    return subprocess.check_output([str(TC / 'Bin' / tool), *map(str,args)], text=True)
for src, obj in [(FW/'src/ichi_ui.c', 'ui.o'), (BUILD/'shim.c', 'shim.o')]:
    run('clang-arm.exe', '-target','arm-none-eabi','-mcpu=cortex-m0plus','-mthumb',
        '-ffreestanding','-fno-builtin','-Os','-I',TC/'Inc/Arm/Clang','-I',TC/'Inc/Arm/CM0p/Picolibc','-I',FW/'include',
        '-c',src,'-o',BUILD/obj)
run('lld-arm.exe','-Ttext=0x10000','--entry=render',BUILD/'ui.o',BUILD/'shim.o','-o',BUILD/'ui.elf')
run('llvm-objcopy.exe','-O','binary',BUILD/'ui.elf',BUILD/'ui.bin')
symbols = {}
for line in run('llvm-readelf.exe','-s',BUILD/'ui.elf').splitlines():
    cols = line.split()
    if len(cols)>=8 and cols[0].rstrip(':').isdigit():
        symbols[cols[-1]] = int(cols[1],16)

tft = (FW/'src/ichi_tft.c').read_text(encoding='utf-8')
def font(name):
    body = re.search(r'\b'+name+r'\[.*?=\s*\{(.*?)\n\};',tft,re.S)[1]
    return [tuple(int(n,16) for n in re.findall(r'0x[0-9a-fA-F]+',row))
            for row in re.findall(r'\{([^}]+)\}',body)]
upper, lower = font('font5x7'), font('font5x7_lower')
assert len(upper)==59 and len(lower)==26
def rgb(c):
    return (round(((c>>11)&31)*255/31),round(((c>>5)&63)*255/63),round((c&31)*255/31))

scenes = ['closing','baseline','ready','listening','transferring','inferring',
          'complete','conditional','failure','hold-add','confirm-add','odl-running',
          'odl-done','confirm-reset','reset-done','canceled','odl-aborted','error','odl-new']
manifest=[]
for index,name in enumerate(scenes):
    im=Image.new('RGB',(320,240)); draw=ImageDraw.Draw(im)
    cpu=Uc(UC_ARCH_ARM,UC_MODE_THUMB)
    cpu.mem_map(0x10000,0x40000);cpu.mem_map(0x20000000,0x10000)
    cpu.mem_write(0x10000,(BUILD/'ui.bin').read_bytes())
    def hook(uc,address,size,data):
        if address not in (symbols['IchiTftFillRect']&~1,symbols['IchiTftDrawText']&~1): return
        x,y,a,b=[uc.reg_read(r) for r in (UC_ARM_REG_R0,UC_ARM_REG_R1,UC_ARM_REG_R2,UC_ARM_REG_R3)]
        sp=uc.reg_read(UC_ARM_REG_SP)
        if address==(symbols['IchiTftFillRect']&~1):
            color=struct.unpack('<I',uc.mem_read(sp,4))[0]&65535
            if a and b: draw.rectangle((x,y,x+a-1,y+b-1),fill=rgb(color))
        else:
            bg,scale=struct.unpack('<II',uc.mem_read(sp,8))
            ptr=a;text=[]
            while (ch:=uc.mem_read(ptr,1)[0]): text.append(chr(ch));ptr+=1
            for ch in text:
                glyph=lower[ord(ch)-97] if 'a'<=ch<='z' else upper[ord(ch)-32 if ' '<=ch<='Z' else 0]
                draw.rectangle((x,y,x+6*scale-1,y+8*scale-1),fill=rgb(bg))
                for col,bits in enumerate(glyph):
                    for row in range(7):
                        if bits&(1<<row): draw.rectangle((x+col*scale,y+row*scale,x+(col+1)*scale-1,y+(row+1)*scale-1),fill=rgb(b))
                x+=6*scale
        uc.reg_write(UC_ARM_REG_PC,uc.reg_read(UC_ARM_REG_LR))
    cpu.hook_add(UC_HOOK_CODE,hook)
    cpu.reg_write(UC_ARM_REG_SP,0x2000FFF0);cpu.reg_write(UC_ARM_REG_LR,0x2F001)
    cpu.reg_write(UC_ARM_REG_R0,index)
    cpu.emu_start(symbols['render']|1,0x2F000,count=2000000)
    path=OUT/(name+'.png');im.save(path)
    manifest.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(OUT/'manifest.json').write_text(json.dumps({'renderer':'Real ichi_ui.c / Cortex-M0+ Unicorn; firmware 08668a8; illustrative state values',
    'ui_sha256':hashlib.sha256((FW/'src/ichi_ui.c').read_bytes()).hexdigest(),'screens':manifest},indent=2)+'\n',encoding='utf-8')
print(f'Rendered {len(scenes)} real-firmware UI screens in {OUT}')

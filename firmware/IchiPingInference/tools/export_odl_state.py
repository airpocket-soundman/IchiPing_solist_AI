"""Run export-only device ODL and verify/export P, beta and baseline over UART."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import time

import numpy as np
import serial
from ichi_serial import decode_frame

def checksum(data):
    h=2166136261
    for b in data:h=((h^b)*16777619)&0xffffffff
    return h

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',default='COM3')
    parser.add_argument('--model',required=True)
    parser.add_argument('--identity-header',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--timeout',type=int,default=7200)
    args=parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    expected=re.search(r'"([A-Fa-f0-9]{192})"',Path(args.identity_header).read_text()).group(1)
    data=bytearray(); meta=None; calibration=None; done=None; seen=set(); steps=set(); buffer=bytearray()
    t0=time.monotonic();last_start=-10
    with serial.Serial(args.port,115200,timeout=.2,write_timeout=3) as port:
        while time.monotonic()-t0<args.timeout and done is None:
            if not seen and time.monotonic()-last_start>3:
                port.write(b'G');last_start=time.monotonic()
            buffer+=port.read(max(1,port.in_waiting))
            while 0 in buffer:
                raw,_,rest=buffer.partition(b'\0');buffer=bytearray(rest)
                fr=decode_frame(bytes(raw)) if raw else None
                if fr is None:continue
                p=fr.payload
                if fr.type==0x40:
                    seen.add('start');print('Device ODL started; no survey',flush=True)
                elif fr.type==0x41:
                    print(f'Baseline {p[0]+1}/3',flush=True)
                elif fr.type==0x45:
                    key=(p[0],p[1])
                    if key in steps:raise RuntimeError('Duplicate ODL update')
                    steps.add(key)
                    print(f'ODL {len(steps)}/160; remaining {160-len(steps)}; elapsed {(time.monotonic()-t0)/60:.1f} min',flush=True)
                elif fr.type==0x46:
                    c,n,ok,seconds=struct.unpack('<HHBH',p)
                    calibration=dict(correct_before=c,samples=n,ok=bool(ok),seconds=seconds)
                    if not ok or n!=160 or len(steps)!=160:raise RuntimeError('Incomplete ODL')
                elif fr.type==0x47:
                    version,hidden,outputs,bins,total,chunks=struct.unpack_from('<HHHHII',p)
                    identity=p[16:].decode('ascii')
                    if (version,hidden,outputs,bins,total,chunks)!=(1,32,32,1024,10240,80):raise RuntimeError('Unexpected export dimensions')
                    if identity!=expected:raise RuntimeError('Firmware model identity mismatch')
                    meta=dict(version=version,identity=identity,total=total,chunks=chunks)
                elif fr.type==0x48:
                    offset=struct.unpack_from('<I',p)[0]
                    if meta is None or offset!=len(data) or len(p)!=132:raise RuntimeError('Out-of-order/truncated export chunk')
                    data+=p[4:]
                elif fr.type==0x49:
                    fnv,total=struct.unpack('<II',p)
                    if len(data)!=10240 or total!=10240 or checksum(data)!=fnv:raise RuntimeError('Export checksum mismatch')
                    seen.add('verified')
                elif fr.type==0x43:
                    done=dict(zip(('correct','total','aborted','seconds'),struct.unpack_from('<HHBH',p)))
                elif fr.type==0x44:raise RuntimeError(f'Device stage error {list(p)}')
                elif fr.type==0x42:raise RuntimeError('Unexpected live survey result')
    if done is None or done['aborted'] or done['total']!=0 or 'verified' not in seen or not calibration:raise RuntimeError('Incomplete ODL export')
    P=np.frombuffer(data[:4096],'<f4').copy().reshape(32,32)
    beta=np.frombuffer(data[4096:8192],'<f4').copy().reshape(32,32)
    baseline=np.frombuffer(data[8192:],'<i2').copy()
    if not np.isfinite(P).all() or not np.isfinite(beta).all():raise RuntimeError('Non-finite ODL weights')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.bin').write_bytes(data)
    model_sha=hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    np.savez_compressed(out.with_suffix('.npz'),P=P,beta=beta,baseline_q=baseline,model_sha256=model_sha,firmware_identity=expected)
    result=dict(meta=meta,calibration=calibration,done=done,model=str(Path(args.model).resolve()),model_sha256=model_sha,
                payload_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),verified=True)
    out.with_suffix('.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE: verified P, beta, baseline and model identity exported',flush=True)

if __name__=='__main__':main()

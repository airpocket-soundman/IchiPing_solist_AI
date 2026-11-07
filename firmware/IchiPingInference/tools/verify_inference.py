"""Wait for startup baseline and request one inference without starting ODL."""
import argparse
import datetime as dt
import json
from pathlib import Path
import struct
import sys
import time
import serial
from ichi_serial import decode_frame

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',default='COM3')
    parser.add_argument('--expect-calibration',choices=('factory','saved','either'),default='either')
    parser.add_argument('--seconds',type=int,default=120)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    start=time.monotonic();buf=bytearray();sent=False;ready=None
    with serial.Serial(args.port,115200,timeout=.2) as port:
        while time.monotonic()-start<args.seconds:
            buf+=port.read(max(1,port.in_waiting))
            while 0 in buf:
                raw,_,rest=buf.partition(b'\0');buf=bytearray(rest)
                frame=decode_frame(bytes(raw)) if raw else None
                if frame is None:continue
                if frame.type==0x50:
                    print(f'Baseline {frame.payload[0]+1}/3',flush=True)
                    if frame.payload[0]==2:ready=time.monotonic()+1
                elif frame.type==0x54:raise RuntimeError(f'Device error {list(frame.payload)}')
                elif frame.type==0x51:
                    actual,pred,cal,exponent,ms=struct.unpack_from('<BBBBI',frame.payload)
                    if args.expect_calibration=='factory' and cal:raise RuntimeError('Unexpected old ODL restoration')
                    if args.expect_calibration=='saved' and not cal:raise RuntimeError('Saved ODL was not restored')
                    result=dict(time=dt.datetime.now().isoformat(),actual=actual,pred=pred,calibrated=bool(cal),
                                correct=actual==pred,transfer_ms=ms,inference_verified=True)
                    Path(args.out).write_text(json.dumps(result,indent=2))
                    print(json.dumps(result),flush=True)
                    return
            if not sent and ((ready is not None and time.monotonic()>=ready) or time.monotonic()-start>60):
                port.write(b'I');sent=True
                print('Inference requested',flush=True)
    raise TimeoutError('No inference result')

if __name__=='__main__':main()

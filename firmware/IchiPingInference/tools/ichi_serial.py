"""IchiPing ファームとの UART 通信 (src/ichi_protocol.c と同じフレーム形式)。

raw frame = "APAN" | version u8 | type u8 | flags u16 | sequence u32 | timestamp u32 | payload_size u16
            | payload | CRC-32 (IEEE, little endian)
wire      = COBS(raw frame) + 0x00
"""
from __future__ import annotations

import struct
import time
import zlib
from dataclasses import dataclass

import numpy as np

HELLO, STATUS, AI_SELFTEST, AI_INFER, AI_RESULT, ACK, NACK = 0x01, 0x02, 0x14, 0x16, 0x21, 0x70, 0x71
VERSION = 1
HEADER = struct.Struct("<4sBBHIIH")
STATUS_FMT = struct.Struct("<BBHBBH")     # version, frontend, input bytes, outputs, cases, last total ms


def cobs_encode(data: bytes) -> bytes:
    out, block = bytearray(), bytearray()
    for b in data:
        if b == 0:
            out += bytes([len(block) + 1]) + block; block.clear()
        else:
            block.append(b)
            if len(block) == 254:
                out += b"\xff" + block; block.clear()
    out += bytes([len(block) + 1]) + block
    return bytes(out)


def cobs_decode(data: bytes) -> bytes:
    out, i = bytearray(), 0
    while i < len(data):
        code = data[i]; i += 1
        if code == 0:
            raise ValueError("zero in COBS data")
        out += data[i:i + code - 1]; i += code - 1
        if code != 0xFF and i < len(data):
            out.append(0)
    return bytes(out)


@dataclass
class Frame:
    type: int
    sequence: int
    payload: bytes


def encode_frame(mtype: int, sequence: int, payload: bytes = b"") -> bytes:
    raw = HEADER.pack(b"APAN", VERSION, mtype, 0, sequence, 0, len(payload)) + payload
    return cobs_encode(raw + struct.pack("<I", zlib.crc32(raw) & 0xFFFFFFFF)) + b"\x00"


def decode_frame(encoded: bytes) -> Frame | None:
    try:
        raw = cobs_decode(encoded)
    except ValueError:
        return None
    if len(raw) < HEADER.size + 4:
        return None
    magic, ver, mtype, _, seq, _, n = HEADER.unpack_from(raw)
    if magic != b"APAN" or ver != VERSION or len(raw) != HEADER.size + n + 4:
        return None
    if struct.unpack_from("<I", raw, len(raw) - 4)[0] != (zlib.crc32(raw[:-4]) & 0xFFFFFFFF):
        return None
    return Frame(mtype, seq, raw[HEADER.size:HEADER.size + n])


class Board:
    def __init__(self, port: str, baud: int = 115200):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0.05, write_timeout=1.0)
        self.buf = bytearray()
        self.seq = 1
        time.sleep(0.2)
        self.ser.reset_input_buffer()

    def request(self, mtype: int, payload: bytes = b"", timeout: float = 10.0) -> Frame:
        seq = self.seq; self.seq += 1
        self.ser.write(encode_frame(mtype, seq, payload))
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.buf += self.ser.read(4096)
            while b"\x00" in self.buf:
                chunk, _, rest = bytes(self.buf).partition(b"\x00")
                self.buf = bytearray(rest)
                fr = decode_frame(chunk) if chunk else None
                if fr is not None and fr.sequence == seq:
                    if fr.type == NACK:
                        raise RuntimeError(f"NACK: request=0x{fr.payload[0]:02X} reason={fr.payload[1]}")
                    return fr
        raise TimeoutError(f"no reply to 0x{mtype:02X} seq={seq}")

    def status(self) -> dict:
        v, fe, nin, nout, ncase, ms = STATUS_FMT.unpack(self.request(STATUS).payload)
        return dict(version=v, frontend=bool(fe), input_bytes=nin, outputs=nout, cases=ncase, last_total_ms=ms)

    def _result(self, fr: Frame, n_out: int):
        if fr.type != AI_RESULT:
            raise RuntimeError(f"unexpected reply 0x{fr.type:02X}")
        case, cls, accel_us, total_ms = struct.unpack_from("<BBHH", fr.payload)
        scores = np.frombuffer(fr.payload, "<f4", count=n_out, offset=6).copy()
        return dict(case=case, cls=cls, accelerator_us=accel_us, total_ms=total_ms, scores=scores)

    def selftest(self, case_id: int, n_out: int) -> dict:
        return self._result(self.request(AI_SELFTEST, bytes([case_id])), n_out)

    def infer(self, payload: bytes, n_out: int) -> dict:
        return self._result(self.request(AI_INFER, payload), n_out)

    def close(self):
        self.ser.close()

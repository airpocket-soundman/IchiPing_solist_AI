"""IchiPing ファームからの UART フレームの復号 (src/ichi_protocol.c と同じフレーム形式)。

raw frame = "APAN" | version u8 | type u8 | flags u16 | sequence u32 | timestamp u32 | payload_size u16
            | payload | CRC-32 (IEEE, little endian)
wire      = COBS(raw frame) + 0x00

メッセージ種別は各ファームの main (src/ichi_*_main.c 冒頭のコメント) を参照。
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

VERSION = 1
HEADER = struct.Struct("<4sBBHIIH")


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

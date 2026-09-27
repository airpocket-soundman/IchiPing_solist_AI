"""Synthesize the PV narration (media/pv/narration.json) with edge-tts (Microsoft neural voice).

    pip install edge-tts
    python media/pv/tts_narration.py          # all lines
    python media/pv/tts_narration.py n16 n17  # only some

Writes media/pv/narration/<id>.mp3. The text is sent to Microsoft's online TTS service.
"""
import asyncio
import json
import sys
from pathlib import Path

import edge_tts

HERE = Path(__file__).resolve().parent


async def main(only):
    spec = json.loads((HERE / "narration.json").read_text(encoding="utf-8"))
    for it in spec["items"]:
        if only and it["id"] not in only:
            continue
        out = HERE / "narration" / f"{it['id']}.mp3"
        await edge_tts.Communicate(it["text"], spec["voice"], rate=it["rate"]).save(str(out))
        print(it["id"], out.name)


asyncio.run(main(set(sys.argv[1:])))

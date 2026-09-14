"""Add concise, manufacturable front-silkscreen pin legends to the THT PCB."""

from pathlib import Path
import sys
import pcbnew


if len(sys.argv) != 3:
    raise SystemExit("usage: add_pin_silks.py <input.kicad_pcb> <output.kicad_pcb>")

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
board = pcbnew.LoadBoard(str(source))


def mm(value: float) -> int:
    return pcbnew.FromMM(value)


def point(x: float, y: float) -> pcbnew.VECTOR2I:
    return pcbnew.VECTOR2I(mm(x), mm(y))


# Text, X, Y, rotation, character height.  Pin numbers follow the top-view
# connector numbering used by CONNECTIONS.md; the square pad remains pin 1.
legends = [
    ("1:SIG 2:GND", 27.2, 23.5, 90, 0.75),  # J_WIN_A
    ("1:SIG 2:GND", 27.2, 32.25, 90, 0.75),  # J_DOOR_AB
    ("1:SIG 2:GND", 27.2, 41.0, 90, 0.75),  # J_WIN_B
    ("1:SIG 2:GND", 27.2, 49.5, 90, 0.75),  # J_DOOR_BC
    ("1:SIG 2:GND", 27.2, 58.25, 90, 0.75),  # J_WIN_C
    ("1:SIG 2:GND", 27.2, 67.0, 90, 0.75),  # J_EXEC
    ("1:RST 2:CS 3:GND 4:5V", 42.0, 24.25, 90, 0.70),
    ("1:NC 2:BL 3:SCK 4:MOSI 5:DC", 38.6, 41.0, 90, 0.70),
    ("1:GND 2:3V3 3:SD 4:BCLK 5:WS 6:GND", 99.3, 35.0, 90, 0.65),
    ("1:WS 2:BCLK 3:DIN 4:GND", 100.8, 51.0, 90, 0.70),
    ("1:SD 2:GND 3:5V", 99.3, 63.0, 90, 0.75),
    ("OPEN IF USB POWERS STAMP", 52.7, 65.5, 0, 0.75),
    ("1:GND 2:SCL 3:SDA 4:3V3", 41.8, 77.0, 0, 0.65),
    ("1:OE 2:GND", 56.5, 77.0, 0, 0.65),
    ("1:5V 2:GND", 71.5, 77.0, 0, 0.65),
    ("1:5V 2:GND", 83.0, 77.0, 0, 0.65),
    ("1:5V 2:GND", 94.0, 77.0, 0, 0.65),
]

# Idempotency: remove only board-level text exactly matching legends created by
# this script.  Footprint references and all user-authored annotations remain.
legend_texts = {entry[0] for entry in legends}
for drawing in list(board.GetDrawings()):
    if (isinstance(drawing, pcbnew.PCB_TEXT)
            and drawing.GetLayer() == pcbnew.F_SilkS
            and drawing.GetText() in legend_texts):
        board.RemoveNative(drawing)

for label, x, y, angle, size in legends:
    text = pcbnew.PCB_TEXT(board)
    text.SetText(label)
    text.SetLayer(pcbnew.F_SilkS)
    text.SetPosition(point(x, y))
    text.SetTextAngle(pcbnew.EDA_ANGLE(angle, pcbnew.DEGREES_T))
    # Project and JLCPCB-friendly minimum character height.
    size = max(size, 0.80)
    text.SetTextSize(pcbnew.VECTOR2I(mm(size), mm(size)))
    text.SetTextThickness(mm(0.13))
    text.SetKeepUpright(False)
    board.Add(text)

destination.parent.mkdir(parents=True, exist_ok=True)
pcbnew.SaveBoard(str(destination), board)
print(f"Added {len(legends)} front-silkscreen pin legends: {destination}")

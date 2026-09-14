"""Prepare the user-arranged THT board for a fresh Specctra route.

The script preserves footprints, drawings, rule areas, board outline and zones.
Only the two Stamp-S3A socket positions, their manufacturing labels, and all
existing tracks/vias are changed.  It writes a new board; the input is never
overwritten.
"""

from pathlib import Path
import sys
import pcbnew


if len(sys.argv) != 4:
    raise SystemExit("usage: reroute_tht.py <input.kicad_pcb> <output.kicad_pcb> <output.dsn>")

input_path = Path(sys.argv[1]).resolve()
output_path = Path(sys.argv[2]).resolve()
dsn_path = Path(sys.argv[3]).resolve()
board = pcbnew.LoadBoard(str(input_path))


def mm(value: float) -> int:
    return pcbnew.FromMM(value)


def point(x: float, y: float) -> pcbnew.VECTOR2I:
    return pcbnew.VECTOR2I(mm(x), mm(y))


def footprint(reference: str) -> pcbnew.FOOTPRINT:
    item = board.FindFootprintByReference(reference)
    if item is None:
        raise RuntimeError(f"Missing footprint: {reference}")
    return item


# Official Stamp-S3A pad-row spacing is 15.24 mm.  These coordinates retain
# the user's 180-degree USB-outward orientation and center the 18 mm-wide body
# outline at X=62 mm.  The 17-pin row is on the right in the carrier top view.
stamp_17 = footprint("J_STAMP_17")
stamp_6 = footprint("J_STAMP_6")
stamp_17.SetPosition(point(62.9217, 33.1485))
stamp_17.SetOrientationDegrees(180.0)
stamp_6.SetPosition(point(47.6817, 25.5285))
stamp_6.SetOrientationDegrees(180.0)

# Keep the selected PCBA/THT part ratings consistent with the schematic/BOM.
footprint("C1").SetValue("470uF/16V LOGIC")
footprint("C3").SetValue("1000uF/35V SERVO")

# Pad 1 moved from the USB end to the antenna end when the module was rotated.
for drawing in board.GetDrawings():
    if isinstance(drawing, pcbnew.PCB_TEXT) and drawing.GetText() == "M1-1":
        drawing.SetPosition(point(72.0, 33.15))
        break
else:
    raise RuntimeError("M1-1 silkscreen label not found")

# The saved placement invalidates every old route.  Remove only tracks and
# vias; do not reconstruct the board or touch the user's component placement.
removed = len(list(board.GetTracks()))
for track in list(board.GetTracks()):
    board.Remove(track)

pcbnew.ZONE_FILLER(board).Fill(board.Zones())
output_path.parent.mkdir(parents=True, exist_ok=True)
dsn_path.parent.mkdir(parents=True, exist_ok=True)
pcbnew.SaveBoard(str(output_path), board)
if not pcbnew.ExportSpecctraDSN(board, str(dsn_path)):
    raise RuntimeError(f"Failed to export DSN: {dsn_path}")

print(f"Prepared {output_path}")
print(f"Removed tracks/vias: {removed}")
print("J_STAMP_17: 62.9217, 33.1485, 180 deg")
print("J_STAMP_6:  47.6817, 25.5285, 180 deg")
print(f"Exported {dsn_path}")

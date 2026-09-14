"""Create a 92 x 69 mm routable copy of the user-arranged THT board.

The latest user-arranged connector/component placement is retained.  Only the
four mounting holes and board-dependent labels move.  The formerly separate
logic and servo 5 V nets are merged before routing.
"""

from pathlib import Path
import sys
import pcbnew


if len(sys.argv) != 4:
    raise SystemExit("usage: compact_tht.py <input.kicad_pcb> <output.kicad_pcb> <output.dsn>")

input_path = Path(sys.argv[1]).resolve()
output_path = Path(sys.argv[2]).resolve()
dsn_path = Path(sys.argv[3]).resolve()
board = pcbnew.LoadBoard(str(input_path))
print("stage: loaded", flush=True)


def mm(value: float) -> int:
    return pcbnew.FromMM(value)


def point(x: float, y: float) -> pcbnew.VECTOR2I:
    return pcbnew.VECTOR2I(mm(x), mm(y))


def move(reference: str, x: float, y: float) -> None:
    item = board.FindFootprintByReference(reference)
    if item is None:
        raise RuntimeError(f"Missing footprint: {reference}")
    item.SetPosition(point(x, y))


# Preserve every electrical part in the user's latest layout.  Move only the
# mounting holes so their centres remain 4 mm from the new board edges.
for ref, x, y in [
    ("H1", 22.0, 14.0),
    ("H2", 106.0, 14.0),
    ("H3", 22.0, 75.0),
    ("H4", 106.0, 75.0),
]:
    move(ref, x, y)
print("stage: footprints", flush=True)

# One common 5 V rail.  Both input connectors remain in parallel so a single
# suitably rated supply can feed the board with adequate current capacity.
logic_5v = board.FindNet("+5V_LOGIC")
servo_5v = board.FindNet("+5V_SERVO")
if logic_5v is None:
    raise RuntimeError("Missing +5V_LOGIC net")
if servo_5v is not None:
    for footprint_item in board.GetFootprints():
        for pad in footprint_item.Pads():
            if pad.GetNetname() == "+5V_SERVO":
                pad.SetNet(logic_5v)
# The common rail carries the five-servo branch as well as the logic/audio
# loads.  Freerouting resolves the existing +5V_LOGIC assignment before any
# later override, so raise its original class width directly.
power_5v = board.GetDesignSettings().m_NetSettings.GetNetClassByName("Power5V")
if power_5v is None:
    raise RuntimeError("Missing Power5V net class")
power_5v.SetTrackWidth(mm(1.0))


# Move the left/right/bottom outline inward; the USB-facing top edge remains.
for drawing in board.GetDrawings():
    if not isinstance(drawing, pcbnew.PCB_SHAPE) or drawing.GetLayer() != pcbnew.Edge_Cuts:
        continue
    start = drawing.GetStart()
    end = drawing.GetEnd()
    sx, sy = pcbnew.ToMM(start.x), pcbnew.ToMM(start.y)
    ex, ey = pcbnew.ToMM(end.x), pcbnew.ToMM(end.y)
    if sx == 10.0:
        sx = 18.0
    elif sx == 120.0:
        sx = 110.0
    if ex == 10.0:
        ex = 18.0
    elif ex == 120.0:
        ex = 110.0
    if sy == 82.0:
        sy = 79.0
    if ey == 82.0:
        ey = 79.0
    drawing.SetStart(point(sx, sy))
    drawing.SetEnd(point(ex, ey))
print("stage: outline", flush=True)


# Move the existing GND-zone outline points in place.  Mutating the original
# line chains preserves KiCad's zone identity and connectivity metadata.
for zone in board.Zones():
    corners = [zone.GetCornerPosition(i) for i in range(zone.GetNumCorners())]
    if not any(corner.x in (mm(10.5), mm(119.5)) for corner in corners):
        continue
    chain = zone.Outline().Outline(0)
    for index, corner in enumerate(corners):
        x = pcbnew.ToMM(corner.x)
        y = pcbnew.ToMM(corner.y)
        if x == 10.5:
            x = 18.5
        elif x == 119.5:
            x = 109.5
        if y == 81.5:
            y = 78.5
        chain.SetPoint(index, point(x, y))
print("stage: zones", flush=True)


# Keep the manually added value labels readable beside their moved parts.
label_positions = {
    "C3 1000uF": (99.5, 60.0),
    "M1-1": (65.3, 33.15),
    "R8 10k": (51.5, 51.0),
    "IchiPing Solist-AI / Stamp-S3A Interposer Rev.B THT": (64.0, 77.2),
}
for drawing in board.GetDrawings():
    if isinstance(drawing, pcbnew.PCB_TEXT) and drawing.GetText() in label_positions:
        drawing.SetPosition(point(*label_positions[drawing.GetText()]))
        if drawing.GetText() == "C3 1000uF":
            drawing.SetTextAngle(pcbnew.EDA_ANGLE(90.0, pcbnew.DEGREES_T))
        elif drawing.GetText().startswith("IchiPing Solist-AI"):
            drawing.SetLayer(pcbnew.B_SilkS)
            drawing.SetMirrored(True)
print("stage: labels", flush=True)

# The Stamp-S3A body outline crosses the USB-facing board edge by design.
# Keep it as an assembly aid on Dwgs.User instead of generating clipped front
# silkscreen.  The actual socket references remain on F.SilkS.
stamp_rect = {(46.3017, 9.9885), (64.3017, 9.9885),
              (46.3017, 35.9885), (64.3017, 35.9885)}
for drawing in board.GetDrawings():
    if not isinstance(drawing, pcbnew.PCB_SHAPE) or drawing.GetLayer() != pcbnew.F_SilkS:
        continue
    start = drawing.GetStart()
    end = drawing.GetEnd()
    endpoints = {
        (round(pcbnew.ToMM(start.x), 4), round(pcbnew.ToMM(start.y), 4)),
        (round(pcbnew.ToMM(end.x), 4), round(pcbnew.ToMM(end.y), 4)),
    }
    if endpoints.issubset(stamp_rect):
        drawing.SetLayer(pcbnew.Dwgs_User)

# The two XH power inputs are electrically parallel.  Put the warning on the
# rear silkscreen where it does not compete with the dense connector labels.
warning_text = "J_PWR_LOGIC + J_PWR_SERVO: SAME 5V SUPPLY ONLY"
if not any(isinstance(item, pcbnew.PCB_TEXT) and item.GetText() == warning_text
           for item in board.GetDrawings()):
    warning = pcbnew.PCB_TEXT(board)
    warning.SetText(warning_text)
    warning.SetLayer(pcbnew.B_SilkS)
    warning.SetPosition(point(64.0, 70.0))
    warning.SetTextSize(pcbnew.VECTOR2I(mm(1.0), mm(1.0)))
    warning.SetTextThickness(mm(0.18))
    warning.SetMirrored(True)
    board.Add(warning)
print("stage: silk", flush=True)


# Every old trace was routed to the former coordinates.  Remove traces/vias
# from the temporary copy and let Freerouting produce a coherent fresh route.
removed = len(list(board.GetTracks()))
for track in list(board.GetTracks()):
    # RemoveNative avoids SWIG attempting to free an object already owned by
    # the board, which can crash KiCad's bundled Python on Windows.
    board.RemoveNative(track)
print("stage: tracks", flush=True)

pcbnew.ZONE_FILLER(board).Fill(board.Zones())
print("stage: filled", flush=True)
output_path.parent.mkdir(parents=True, exist_ok=True)
pcbnew.SaveBoard(str(output_path), board)
if not pcbnew.ExportSpecctraDSN(board, str(dsn_path)):
    raise RuntimeError(f"Failed to export DSN: {dsn_path}")

print(f"Prepared compact board: {output_path}")
print(f"Removed tracks/vias: {removed}")
print("Board size: 92 x 69 mm")
print(f"Exported: {dsn_path}")

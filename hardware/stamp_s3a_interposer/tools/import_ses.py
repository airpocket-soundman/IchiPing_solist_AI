"""Import a Specctra session into a KiCad board and refill all zones."""

from pathlib import Path
import sys
import pcbnew


if len(sys.argv) != 4:
    raise SystemExit("usage: import_ses.py <input.kicad_pcb> <input.ses> <output.kicad_pcb>")

board_path = Path(sys.argv[1]).resolve()
session_path = Path(sys.argv[2]).resolve()
output_path = Path(sys.argv[3]).resolve()
board = pcbnew.LoadBoard(str(board_path))
if not pcbnew.ImportSpecctraSES(board, str(session_path)):
    raise RuntimeError(f"Failed to import Specctra session: {session_path}")

# Refill once to expose any small GND islands created by the routed signal
# channels. Stitch each island to copper on the opposite layer wherever both
# pours overlap with at least 0.5 mm of copper around the via centre.
pcbnew.ZONE_FILLER(board).Fill(board.Zones())
gnd = board.FindNet("GND")
gnd_zones = {
    zone.GetLayer(): zone
    for zone in board.Zones()
    if zone.GetNetname() == "GND" and zone.GetLayer() in (pcbnew.F_Cu, pcbnew.B_Cu)
}
if set(gnd_zones) != {pcbnew.F_Cu, pcbnew.B_Cu}:
    raise RuntimeError("Expected one GND zone on F.Cu and B.Cu")


def contains_with_margin(shape, x: float, y: float, margin: float = 0.5) -> bool:
    offsets = ((0, 0), (margin, 0), (-margin, 0), (0, margin), (0, -margin))
    def contains(test_point: pcbnew.VECTOR2I) -> bool:
        if hasattr(shape, "Contains"):
            return shape.Contains(test_point)
        return shape.PointInside(test_point)
    return all(contains(pcbnew.VECTOR2I(pcbnew.FromMM(x + dx), pcbnew.FromMM(y + dy)))
               for dx, dy in offsets)


drills: list[tuple[float, float, float]] = []
for footprint in board.GetFootprints():
    for pad in footprint.Pads():
        drill = pad.GetDrillSize()
        if drill.x > 0:
            position = pad.GetPosition()
            drills.append((pcbnew.ToMM(position.x), pcbnew.ToMM(position.y),
                           max(pcbnew.ToMM(drill.x), pcbnew.ToMM(drill.y)) / 2))
for track in board.GetTracks():
    if isinstance(track, pcbnew.PCB_VIA):
        position = track.GetPosition()
        drills.append((pcbnew.ToMM(position.x), pcbnew.ToMM(position.y),
                       pcbnew.ToMM(track.GetDrillValue()) / 2))

stitches: list[tuple[float, float]] = []
for layer, zone in gnd_zones.items():
    own = zone.GetFilledPolysList(layer)
    other_layer = pcbnew.B_Cu if layer == pcbnew.F_Cu else pcbnew.F_Cu
    other = gnd_zones[other_layer].GetFilledPolysList(other_layer)
    # Outline zero is the large connected pour; subsequent outlines are the
    # isolated copper regions that need a stitching via.
    for island_index in range(1, own.OutlineCount()):
        island = own.Outline(island_index)
        bbox = island.BBox()
        found = None
        x0 = pcbnew.ToMM(bbox.GetX()) + 0.75
        x1 = pcbnew.ToMM(bbox.GetRight()) - 0.75
        y0 = pcbnew.ToMM(bbox.GetY()) + 0.75
        y1 = pcbnew.ToMM(bbox.GetBottom()) - 0.75
        x = x0
        while x <= x1 and found is None:
            y = y0
            while y <= y1:
                if (contains_with_margin(island, x, y) and
                        contains_with_margin(other, x, y) and
                        all((x - sx) ** 2 + (y - sy) ** 2 >= 2.0 ** 2
                            for sx, sy in stitches) and
                        all((x - dx) ** 2 + (y - dy) ** 2 >= (radius + 0.45) ** 2
                            for dx, dy, radius in drills)):
                    found = (x, y)
                    break
                y += 0.5
            x += 0.5
        if found is not None:
            stitches.append(found)

for x, y in stitches:
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y)))
    via.SetWidth(pcbnew.FromMM(0.8))
    via.SetDrill(pcbnew.FromMM(0.4))
    via.SetViaType(pcbnew.VIATYPE_THROUGH)
    via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
    via.SetNet(gnd)
    board.Add(via)

pcbnew.ZONE_FILLER(board).Fill(board.Zones())
pcbnew.SaveBoard(str(output_path), board)
print(f"Imported route: {output_path}")
print(f"GND stitching vias: {len(stitches)}")

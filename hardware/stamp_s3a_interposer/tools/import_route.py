"""Import a Freerouting SES file and normalize KiCad manufacturing widths."""

from pathlib import Path
import sys
import pcbnew

HERE = Path(__file__).resolve().parent.parent
if len(sys.argv) < 3:
    raise SystemExit("usage: import_route.py <board.kicad_pcb> <route.ses>")
BOARD_PATH = Path(sys.argv[1])
SES_PATH = Path(sys.argv[2])

board = pcbnew.LoadBoard(str(BOARD_PATH))
if not pcbnew.ImportSpecctraSES(board, str(SES_PATH)):
    raise SystemExit(f"Failed to import {SES_PATH}")


def mm(value: float) -> int:
    return pcbnew.FromMM(value)


def point(x: float, y: float) -> pcbnew.VECTOR2I:
    return pcbnew.VECTOR2I(mm(x), mm(y))


def add_ground_zone(layer: int) -> None:
    zone = pcbnew.ZONE(board)
    zone.SetLayer(layer)
    zone.SetNet(board.FindNet("GND"))
    zone.SetLocalClearance(mm(0.30))
    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
    outline = zone.Outline()
    outline.NewOutline()
    for x, y in [
        (10.5, 10.5), (52.0, 10.5), (52.0, 18.0), (70.0, 18.0),
        (70.0, 10.5), (119.5, 10.5), (119.5, 81.5), (10.5, 81.5),
    ]:
        outline.Append(point(x, y))
    board.Add(zone)

for item in board.GetTracks():
    if isinstance(item, pcbnew.PCB_VIA):
        item.SetWidth(pcbnew.FromMM(0.80))
        item.SetDrill(pcbnew.FromMM(0.40))
        continue
    name = item.GetNetname()
    if name == "+5V_LOGIC":
        width = 1.00
    elif name in {"+5V_STAMP", "+3V3_STAMP", "GND"}:
        width = 0.20 if name == "GND" else 0.30
    else:
        width = 0.20
    item.SetWidth(pcbnew.FromMM(width))

zones = list(board.Zones())
ground_zones = [zone for zone in zones if not zone.GetIsRuleArea() and zone.GetNetname() == "GND"]
if ground_zones:
    for zone in ground_zones:
        zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
        zone.SetLocalClearance(mm(0.30))
else:
    add_ground_zone(pcbnew.F_Cu)
    add_ground_zone(pcbnew.B_Cu)
pcbnew.ZONE_FILLER(board).Fill(board.Zones())

pcbnew.SaveBoard(str(BOARD_PATH), board)
print(f"Imported {SES_PATH} and saved {BOARD_PATH}")

"""Merge the legacy servo 5 V net into the common logic/audio 5 V net."""

from pathlib import Path
import sys
import pcbnew


if len(sys.argv) != 3:
    raise SystemExit("usage: merge_common_5v.py <input.kicad_pcb> <output.kicad_pcb>")

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
board = pcbnew.LoadBoard(str(source))
common = board.FindNet("+5V_LOGIC")
servo = board.FindNet("+5V_SERVO")
if common is None or servo is None:
    raise RuntimeError("Expected +5V_LOGIC and +5V_SERVO nets")

for footprint in board.GetFootprints():
    for pad in footprint.Pads():
        if pad.GetNetname() == "+5V_SERVO":
            pad.SetNet(common)

for track in board.GetTracks():
    if track.GetNetname() == "+5V_SERVO":
        track.SetNet(common)

for zone in board.Zones():
    if zone.GetNetname() == "+5V_SERVO":
        zone.SetNet(common)

# New routing uses a 1 mm minimum.  Existing narrower logic branches remain
# geometrically unchanged here to avoid silently creating clearance errors in
# the already-routed PCBA layout; they must not be used as servo-current paths.
power_5v = board.GetDesignSettings().m_NetSettings.GetNetClassByName("Power5V")
if power_5v is not None:
    power_5v.SetTrackWidth(pcbnew.FromMM(1.0))

pcbnew.ZONE_FILLER(board).Fill(board.Zones())
destination.parent.mkdir(parents=True, exist_ok=True)
pcbnew.SaveBoard(str(destination), board)
print(f"Merged common 5 V rail: {destination}")

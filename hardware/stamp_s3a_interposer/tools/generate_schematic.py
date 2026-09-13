"""Regenerate the THT schematic from the parity-clean common topology.

The PCBA and THT boards have the same references, pin numbers, values and
nets; only R1-R9 and C2/C4 footprints differ.  The PCBA schematic contains
the reviewed custom connector symbols and intentional-NC representation that
KiCad's schematic-parity checker accepts.  This script clones that topology,
changes only the project identity and THT footprints, then verifies the result.
It never writes the PCBA/common schematic.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent.parent
SOURCE = HERE / "stamp_s3a_interposer_pcba.kicad_sch"
TARGET = HERE / "stamp_s3a_interposer_tht.kicad_sch"
BOARD = HERE / "stamp_s3a_interposer_tht.kicad_pcb"
REPORTS = HERE / "build" / "reports"
KICAD = Path(r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe")
text = SOURCE.read_text(encoding="utf-8")

replacements = {
    "stamp_s3a_interposer_pcba": "stamp_s3a_interposer_tht",
    "Interposer - PCBA": "Interposer - THT",
    "Resistor_SMD:R_0603_1608Metric": "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal",
    "Capacitor_SMD:C_0603_1608Metric": "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm",
}
for old, new in replacements.items():
    text = text.replace(old, new)

# Use resolvable library-qualified footprints. The PCB is updated to the same
# FPIDs below, preserving geometry, pad numbers, nets and routing.
footprints = {
    "C_Disc_D5.0mm_W2.5mm_P5.00mm": "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm",
    "CP_Radial_D10.0mm_P5.00mm": "Capacitor_THT:CP_Radial_D10.0mm_P5.00mm",
    "CP_Radial_D12.5mm_P5.00mm": "Capacitor_THT:CP_Radial_D12.5mm_P5.00mm",
    "CP_Radial_D5.0mm_P2.00mm": "Capacitor_THT:CP_Radial_D5.0mm_P2.00mm",
    "JST_XH_B2B-XH-A_1x02_P2.50mm_Vertical": "Connector_JST:JST_XH_B2B-XH-A_1x02_P2.50mm_Vertical",
    "JST_XH_B3B-XH-A_1x03_P2.50mm_Vertical": "Connector_JST:JST_XH_B3B-XH-A_1x03_P2.50mm_Vertical",
    "JST_XH_B4B-XH-A_1x04_P2.50mm_Vertical": "Connector_JST:JST_XH_B4B-XH-A_1x04_P2.50mm_Vertical",
    "JST_XH_B5B-XH-A_1x05_P2.50mm_Vertical": "Connector_JST:JST_XH_B5B-XH-A_1x05_P2.50mm_Vertical",
    "JST_XH_B6B-XH-A_1x06_P2.50mm_Vertical": "Connector_JST:JST_XH_B6B-XH-A_1x06_P2.50mm_Vertical",
    "MountingHole_3.2mm_M3": "MountingHole:MountingHole_3.2mm_M3",
    "PinHeader_1x02_P2.54mm_Vertical": "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
    "PinSocket_1x06_P2.54mm_Vertical": "Connector_PinSocket_2.54mm:PinSocket_1x06_P2.54mm_Vertical",
    "PinSocket_1x17_P1.27mm_Vertical": "Connector_PinSocket_1.27mm:PinSocket_1x17_P1.27mm_Vertical",
    "PinSocket_2x07_P2.54mm_Vertical": "Connector_PinSocket_2.54mm:PinSocket_2x07_P2.54mm_Vertical",
    "R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal": "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal",
}
for bare, qualified in footprints.items():
    text = text.replace(f'(property "Footprint" "{bare}"', f'(property "Footprint" "{qualified}"')

# Keep the long microphone pin-order note inside the audio block at 150 dpi.
text = text.replace('(at 24 160 0.0000)', '(at 56 160 0.0000)', 1)
text = text.replace('(at 24 166 0.0000)', '(at 36 166 0.0000)', 1)
text = text.replace('(at 126 89 0.0000)', '(at 145 89 0.0000)', 1)

TARGET.write_text(text, encoding="utf-8", newline="\n")

board_text = BOARD.read_text(encoding="utf-8")
for bare, qualified in footprints.items():
    board_text = board_text.replace(f'(footprint "{bare}"', f'(footprint "{qualified}"')
board_text = board_text.replace(
    '(footprint "Connector_PinSocket_2.54mm:PinSocket_1x06_P2.54mm_Vertical"',
    '(footprint "IchiPing_Interposer:Stamp_S3A_Right_Socket_1x06_P2.54mm"')
BOARD.write_text(board_text, encoding="utf-8", newline="\n")

REPORTS.mkdir(parents=True, exist_ok=True)
subprocess.run([
    str(KICAD), "sch", "erc", "--severity-all",
    "-o", str(REPORTS / "erc_tht.rpt"), str(TARGET),
], check=True)
subprocess.run([
    str(KICAD), "pcb", "drc", "--schematic-parity", "--severity-all",
    "-o", str(REPORTS / "drc_tht_parity.rpt"), str(BOARD),
], check=True)
print(f"Generated {TARGET}")
print("Parity report must end with: Found 0 Footprint errors")

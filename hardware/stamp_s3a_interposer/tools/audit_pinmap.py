"""Fail if a compatibility-critical connector pad is on the wrong net."""

from pathlib import Path
import pcbnew


HERE = Path(__file__).resolve().parent.parent
EXPECTED = {
    "J_SOLIST": ["I2C_SCL", "GND", "I2C_SDA", "GND", "", "TFT_RST_N", "TFT_DC", "", "TFT_MOSI", "PCA_OE", "GND", "TFT_SCK", "GND", "TFT_CS_N"],
    "J_WIN_A": ["SW_WIN_A", "GND"],
    "J_WIN_B": ["SW_WIN_B", "GND"],
    "J_WIN_C": ["SW_WIN_C", "GND"],
    "J_DOOR_AB": ["SW_DOOR_AB", "GND"],
    "J_DOOR_BC": ["SW_DOOR_BC", "GND"],
    "J_EXEC": ["EXEC_N", "GND"],
    "J_MIC": ["GND", "+3V3_STAMP", "I2S_DIN_MIC", "I2S_BCLK_MIC", "I2S_WS_MIC", "GND"],
    "J_AMP_SIG": ["I2S_WS_AMP", "I2S_BCLK_AMP", "I2S_DOUT_AMP", "GND"],
    "J_AMP_PWR": ["AMP_SD", "GND", "+5V_LOGIC"],
    "J_TFT_SIG": ["", "+5V_LOGIC", "TFT_SCK", "TFT_MOSI", "TFT_DC"],
    "J_TFT_PWR": ["TFT_RST_N", "TFT_CS_N", "GND", "+5V_LOGIC"],
    "J_SERVO_CTRL": ["GND", "I2C_SCL", "I2C_SDA", "+3V3_STAMP"],
    "J_PCA_OE": ["PCA_OE", "GND"],
    "J_SERVO_5V_OUT": ["+5V_LOGIC", "GND"],
    "J_PWR_LOGIC": ["+5V_LOGIC", "GND"],
    "J_PWR_SERVO": ["+5V_LOGIC", "GND"],
}

STAMP_17_EXPECTED = {
    1: "AMP_SD", 2: "SW_WIN_B", 3: "", 4: "SW_WIN_C",
    5: "I2S_WS_SRC", 6: "SW_DOOR_AB", 7: "I2S_DOUT_SRC",
    8: "SW_DOOR_BC", 9: "I2S_DIN_MIC", 10: "EXEC_N", 11: "GND",
    12: "", 13: "+5V_STAMP", 14: "", 15: "I2C_SDA", 16: "",
    17: "I2C_SCL",
}
STAMP_6_EXPECTED = {
    18: "GND", 20: "", 22: "", 24: "SW_WIN_A",
    26: "I2S_BCLK_SRC", 28: "+3V3_STAMP",
}

failures: list[str] = []
for variant in ("pcba", "tht"):
    board_path = HERE / f"stamp_s3a_interposer_{variant}.kicad_pcb"
    board = pcbnew.LoadBoard(str(board_path))
    for reference, nets in EXPECTED.items():
        footprint = board.FindFootprintByReference(reference)
        if footprint is None:
            failures.append(f"{variant}: missing {reference}")
            continue
        for number, expected in enumerate(nets, 1):
            pad = footprint.FindPadByNumber(str(number))
            actual = pad.GetNetname() if pad else "<missing pad>"
            if actual != expected:
                failures.append(f"{variant}: {reference}-{number}: expected {expected!r}, got {actual!r}")

    for reference, mapping in (("J_STAMP_17", STAMP_17_EXPECTED),
                               ("J_STAMP_6", STAMP_6_EXPECTED)):
        footprint = board.FindFootprintByReference(reference)
        if footprint is None:
            failures.append(f"{variant}: missing {reference}")
            continue
        for number, expected in mapping.items():
            pad = footprint.FindPadByNumber(str(number))
            actual = pad.GetNetname() if pad else "<missing pad>"
            if actual != expected:
                failures.append(f"{variant}: {reference}-{number}: expected {expected!r}, got {actual!r}")

    # Carrier top view is intentionally X-mirrored per the actual plug-in
    # orientation: M1-1/right, M1-18/left.  This prevents back-side-up fit.
    p1 = board.FindFootprintByReference("J_STAMP_17").FindPadByNumber("1").GetPosition()
    p18 = board.FindFootprintByReference("J_STAMP_6").FindPadByNumber("18").GetPosition()
    if not p1.x > p18.x:
        failures.append(f"{variant}: Stamp socket is not X-mirrored (M1-1 must be right of M1-18)")

if failures:
    raise SystemExit("PINMAP AUDIT FAILED\n" + "\n".join(failures))
print("PINMAP AUDIT PASS: PCBA and THT variants; external connectors and mirrored Stamp sockets")

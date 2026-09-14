"""Generate the KiCad 10 carrier PCB from a reviewed pin map.

Run with KiCad's bundled Python, not the system Python:
  & 'C:\Program Files\KiCad\10.0\bin\python.exe' tools/generate_board.py

The generated PCB is intentionally reproducible.  Connector pin numbers are
the compatibility contract; do not reorder them to make routing prettier.
"""

from __future__ import annotations

from pathlib import Path
import os
import pcbnew


HERE = Path(__file__).resolve().parent.parent
KICAD = Path(r"C:\Program Files\KiCad\10.0\share\kicad\footprints")
LOCAL = HERE / "lib" / "IchiPing_Interposer.pretty"
VARIANT = os.environ.get("BOARD_VARIANT", "pcba").lower()
if VARIANT not in {"pcba", "tht"}:
    raise SystemExit("BOARD_VARIANT must be 'pcba' or 'tht'")
OUT = HERE / f"stamp_s3a_interposer_{VARIANT}.kicad_pcb"


def mm(value: float) -> int:
    return pcbnew.FromMM(value)


def point(x: float, y: float) -> pcbnew.VECTOR2I:
    return pcbnew.VECTOR2I(mm(x), mm(y))


board = pcbnew.BOARD()
settings = board.GetDesignSettings()
settings.SetCopperLayerCount(2)
settings.m_TrackMinWidth = mm(0.20)
settings.m_ViasMinSize = mm(0.80)
settings.m_ViasMinDrill = mm(0.40)


def add_netclass(name: str, track: float, via: float = 0.80,
                 drill: float = 0.40, clearance: float = 0.20) -> None:
    cls = pcbnew.NETCLASS(name)
    cls.SetTrackWidth(mm(track))
    cls.SetViaDiameter(mm(via))
    cls.SetViaDrill(mm(drill))
    cls.SetClearance(mm(clearance))
    settings.m_NetSettings.SetNetclass(name, cls)


default_class = settings.m_NetSettings.GetNetClassByName("Default")
default_class.SetTrackWidth(mm(0.20))
default_class.SetViaDiameter(mm(0.80))
default_class.SetViaDrill(mm(0.40))
default_class.SetClearance(mm(0.20))
add_netclass("Power3V3", 0.30)
add_netclass("Power5V", 1.00)
add_netclass("ServoPower", 1.00)
settings.m_NetSettings.SetNetclassPatternAssignment("+3V3_STAMP", "Power3V3")
settings.m_NetSettings.SetNetclassPatternAssignment("+5V_LOGIC", "Power5V")
settings.m_NetSettings.SetNetclassPatternAssignment("+5V_STAMP", "Power5V")


NETS: dict[str, pcbnew.NETINFO_ITEM] = {}


def net(name: str) -> pcbnew.NETINFO_ITEM:
    if name not in NETS:
        item = pcbnew.NETINFO_ITEM(board, name)
        board.Add(item)
        NETS[name] = item
    return NETS[name]


def load(lib: str, name: str) -> pcbnew.FOOTPRINT:
    root = LOCAL if lib == "LOCAL" else KICAD / f"{lib}.pretty"
    fp = pcbnew.FootprintLoad(str(root), name)
    if fp is None:
        raise RuntimeError(f"Footprint not found: {root}/{name}")
    nickname = "IchiPing_Interposer" if lib == "LOCAL" else lib
    fp.SetFPID(pcbnew.LIB_ID(f"{nickname}:{name}"))
    return fp


def add_fp(ref: str, value: str, lib: str, name: str, x: float, y: float,
           rotation: float = 0, pins: dict[int, str] | None = None) -> pcbnew.FOOTPRINT:
    fp = load(lib, name)
    fp.SetReference(ref)
    fp.SetValue(value)
    fp.SetPosition(point(x, y))
    fp.SetOrientationDegrees(rotation)
    board.Add(fp)
    for pad_number, net_name in (pins or {}).items():
        pad = fp.FindPadByNumber(str(pad_number))
        if pad is None:
            raise RuntimeError(f"{ref} has no pad {pad_number}")
        pad.SetNet(net(net_name))
    return fp


XH = {
    2: "JST_XH_B2B-XH-A_1x02_P2.50mm_Vertical",
    3: "JST_XH_B3B-XH-A_1x03_P2.50mm_Vertical",
    4: "JST_XH_B4B-XH-A_1x04_P2.50mm_Vertical",
    5: "JST_XH_B5B-XH-A_1x05_P2.50mm_Vertical",
    6: "JST_XH_B6B-XH-A_1x06_P2.50mm_Vertical",
}


def xh(ref: str, value: str, count: int, x: float, y: float, rotation: float,
       names: list[str]) -> pcbnew.FOOTPRINT:
    assert len(names) == count
    return add_fp(ref, value, "Connector_JST", XH[count], x, y, rotation,
                  {index + 1: name for index, name in enumerate(names) if name})


# Board-to-board interfaces.  The Solist connector is a 2x7 MIL-style socket;
# numbering is odd/even by row and matches CN3/CN1 in the board manual.
solist = add_fp(
    "J_SOLIST", "SOLIST_CN3_2x7", "Connector_PinSocket_2.54mm",
    "PinSocket_2x07_P2.54mm_Vertical", 24, 53, 0,
    {
        1: "I2C_SCL", 2: "GND", 3: "I2C_SDA", 4: "GND",
        6: "TFT_RST_N", 7: "TFT_DC", 9: "TFT_MOSI",
        10: "PCA_OE", 11: "GND", 12: "TFT_SCK", 13: "GND",
        14: "TFT_CS_N",
    },
)

def stamp_socket(ref: str, count: int, pitch: float, x: float, y: float,
                 m1_numbers: list[int], pins: dict[int, str]) -> pcbnew.FOOTPRINT:
    pitch_text = "1.27" if pitch == 1.27 else "2.54"
    if ref == "J_STAMP_6":
        fp = load("LOCAL", "Stamp_S3A_Right_Socket_1x06_P2.54mm")
        original_pads = [fp.FindPadByNumber(str(number)) for number in m1_numbers]
    else:
        fp = load(f"Connector_PinSocket_{pitch_text}mm",
                  f"PinSocket_1x{count:02d}_P{pitch_text}mm_Vertical")
        original_pads = [fp.FindPadByNumber(str(i)) for i in range(1, count + 1)]
    fp.SetReference(ref)
    fp.SetValue(f"Stamp-S3A_{count}P_SOCKET")
    fp.SetPosition(point(x, y))
    board.Add(fp)
    for item, m1_number in zip(original_pads, m1_numbers):
        if item is None:
            raise RuntimeError(f"{ref} has no source pad for M1-{m1_number}")
        item.SetNumber(str(m1_number))
        if m1_number in pins:
            item.SetNet(net(pins[m1_number]))
    return fp


# Carrier top view is intentionally X-mirrored for the user's actual header
# installation: M1 1.27-mm/17P is on the right and M1 even/6P is on the left.
# Both are separate references so JLCPCB can place two physical sockets.
stamp_nets = {
    1: "AMP_SD", 2: "SW_WIN_B", 4: "SW_WIN_C", 5: "I2S_WS_SRC",
    6: "SW_DOOR_AB", 7: "I2S_DOUT_SRC", 8: "SW_DOOR_BC",
    9: "I2S_DIN_MIC", 11: "GND", 13: "+5V_STAMP",
    15: "I2C_SDA", 17: "I2C_SCL", 18: "GND",
    10: "EXEC_N", 24: "SW_WIN_A", 26: "I2S_BCLK_SRC", 28: "+3V3_STAMP",
}
stamp_socket("J_STAMP_17", 17, 1.27, 68.62, 11.84,
             list(range(1, 18)), stamp_nets)
stamp_socket("J_STAMP_6", 6, 2.54, 53.38, 19.46,
             [28, 26, 24, 22, 20, 18], stamp_nets)

# External harnesses: genuine JST XH, 2.50 mm pitch.  Pin order intentionally
# follows the working IchiPing UNO Q harness contract.
xh("J_WIN_A", "WIN_A:1=SIG,2=GND", 2, 14, 23, 90, ["SW_WIN_A", "GND"])
xh("J_WIN_B", "WIN_B:1=SIG,2=GND", 2, 14, 33, 90, ["SW_WIN_B", "GND"])
xh("J_WIN_C", "WIN_C:1=SIG,2=GND", 2, 14, 43, 90, ["SW_WIN_C", "GND"])
xh("J_DOOR_AB", "DOOR_AB:1=SIG,2=GND", 2, 14, 53, 90, ["SW_DOOR_AB", "GND"])
xh("J_DOOR_BC", "DOOR_BC:1=SIG,2=GND", 2, 14, 64, 90, ["SW_DOOR_BC", "GND"])
xh("J_EXEC", "EXEC:1=SIG,2=GND", 2, 95, 15, 0, ["EXEC_N", "GND"])

xh("J_MIC", "MIC:GND,3V3,SD,SCK,WS,LR", 6, 108, 26, 90,
   ["GND", "+3V3_STAMP", "I2S_DIN_MIC", "I2S_BCLK_MIC", "I2S_WS_MIC", "GND"])
xh("J_AMP_SIG", "AMP:LRC,BCLK,DIN,GND", 4, 108, 42, 90,
   ["I2S_WS_AMP", "I2S_BCLK_AMP", "I2S_DOUT_AMP", "GND"])
xh("J_AMP_PWR", "AMP:SD,GND,5V", 3, 108, 55, 90,
   ["AMP_SD", "GND", "+5V_LOGIC"])

xh("J_TFT_SIG", "TFT:MISO_NC,BL,SCK,MOSI,DC", 5, 24, 15, 0,
   ["", "+5V_LOGIC", "TFT_SCK", "TFT_MOSI", "TFT_DC"])
xh("J_TFT_PWR", "TFT:RST,CS,GND,VCC", 4, 40, 20, 0,
   ["TFT_RST_N", "TFT_CS_N", "GND", "+5V_LOGIC"])

xh("J_SERVO_CTRL", "PCA:GND,SCL,SDA,3V3", 4, 47, 75, 0,
   ["GND", "I2C_SCL", "I2C_SDA", "+3V3_STAMP"])
xh("J_PCA_OE", "PCA:OE,GND", 2, 61, 75, 0, ["PCA_OE", "GND"])
xh("J_SERVO_5V_OUT", "SERVO:COMMON_5V,GND", 2, 94, 75, 0,
   ["+5V_LOGIC", "GND"])
xh("J_PWR_LOGIC", "INPUT:COMMON_5V,GND", 2, 82, 75, 0,
   ["+5V_LOGIC", "GND"])
xh("J_PWR_SERVO", "INPUT:COMMON_5V,GND", 2, 105, 75, 0,
   ["+5V_LOGIC", "GND"])

# Safety and source termination components.  DIN/GAIN/LR retain the UNO-Q
# connector order while reset defaults are changed for Stamp control safety.
R_LIB, R_FP = (("Resistor_SMD", "R_0603_1608Metric") if VARIANT == "pcba" else
               ("Resistor_THT", "R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal"))
C_LIB, C_FP = (("Capacitor_SMD", "C_0603_1608Metric") if VARIANT == "pcba" else
               ("Capacitor_THT", "C_Disc_D5.0mm_W2.5mm_P5.00mm"))
resistors = [
    ("R1", "33R BCLK_MIC", 79, 18, "I2S_BCLK_SRC", "I2S_BCLK_MIC"),
    ("R2", "33R BCLK_AMP", 79, 24, "I2S_BCLK_SRC", "I2S_BCLK_AMP"),
    ("R3", "33R WS_MIC", 79, 30, "I2S_WS_SRC", "I2S_WS_MIC"),
    ("R4", "33R WS_AMP", 79, 36, "I2S_WS_SRC", "I2S_WS_AMP"),
    ("R5", "33R AMP_DIN", 79, 44, "I2S_DOUT_SRC", "I2S_DOUT_AMP"),
    ("R6", "10k AMP_SD pull-down", 79, 52, "AMP_SD", "GND"),
    ("R7", "100k MIC_SD pull-down", 79, 58, "I2S_DIN_MIC", "GND"),
    ("R8", "10k PCA_OE pull-up", 43 if VARIANT == "pcba" else 55,
     65 if VARIANT == "pcba" else 68, "PCA_OE", "+3V3_STAMP"),
    ("R9", "10k TFT_CS pull-up", 30, 26, "TFT_CS_N", "+3V3_STAMP"),
]
for ref, value, x, y, a, b in resistors:
    add_fp(ref, value, R_LIB, R_FP, x, y, 0, {1: a, 2: b})


add_fp("JP_STAMP_5V", "OPEN WHEN USB POWERS STAMP", "Connector_PinHeader_2.54mm",
       "PinHeader_1x02_P2.54mm_Vertical", 84, 63, 0,
       {1: "+5V_LOGIC", 2: "+5V_STAMP"})

add_fp("C1", "470uF/16V LOGIC", "Capacitor_THT", "CP_Radial_D10.0mm_P5.00mm",
       94, 62, 0, {1: "+5V_LOGIC", 2: "GND"})
add_fp("C2", "100nF LOGIC", C_LIB, C_FP, 70, 66, 0,
       {1: "+5V_LOGIC", 2: "GND"})
add_fp("C3", "1000uF/35V SERVO", "Capacitor_THT", "CP_Radial_D12.5mm_P5.00mm",
       108, 65, 0, {1: "+5V_LOGIC", 2: "GND"})
add_fp("C4", "100nF MIC", C_LIB, C_FP, 98, 28, 0,
       {1: "+3V3_STAMP", 2: "GND"})
add_fp("C5", "10uF AMP", "Capacitor_THT", "CP_Radial_D5.0mm_P2.00mm",
       98, 49, 0, {1: "+5V_LOGIC", 2: "GND"})


def pad(ref: str, number: int) -> pcbnew.PAD:
    fp = board.FindFootprintByReference(ref)
    found = fp.FindPadByNumber(str(number)) if fp else None
    if found is None:
        raise RuntimeError(f"Missing {ref}.{number}")
    return found


def segment(a: pcbnew.VECTOR2I, b: pcbnew.VECTOR2I, net_name: str,
            layer: int, width: float = 0.25) -> None:
    if a == b:
        return
    track = pcbnew.PCB_TRACK(board)
    track.SetStart(a)
    track.SetEnd(b)
    track.SetLayer(layer)
    track.SetWidth(mm(width))
    track.SetNet(net(net_name))
    board.Add(track)


def route(ref_a: str, pin_a: int, ref_b: str, pin_b: int, net_name: str,
          layer: int = pcbnew.F_Cu, width: float = 0.25,
          dogleg: float | None = None) -> None:
    a = pad(ref_a, pin_a).GetPosition()
    b = pad(ref_b, pin_b).GetPosition()
    if dogleg is None:
        mid = pcbnew.VECTOR2I(b.x, a.y)
        segment(a, mid, net_name, layer, width)
        segment(mid, b, net_name, layer, width)
    else:
        lane = mm(dogleg)
        p1 = pcbnew.VECTOR2I(a.x, lane)
        p2 = pcbnew.VECTOR2I(b.x, lane)
        segment(a, p1, net_name, layer, width)
        segment(p1, p2, net_name, layer, width)
        segment(p2, b, net_name, layer, width)


# Region-based routing.  Grounds are connected by the B.Cu plane below.
route("J_STAMP_6", 24, "J_WIN_A", 1, "SW_WIN_A", pcbnew.F_Cu, dogleg=19)
route("J_STAMP_17", 2, "J_WIN_B", 1, "SW_WIN_B", pcbnew.F_Cu, dogleg=30)
route("J_STAMP_17", 4, "J_WIN_C", 1, "SW_WIN_C", pcbnew.F_Cu, dogleg=41)
route("J_STAMP_17", 6, "J_DOOR_AB", 1, "SW_DOOR_AB", pcbnew.F_Cu, dogleg=52)
route("J_STAMP_17", 8, "J_DOOR_BC", 1, "SW_DOOR_BC", pcbnew.F_Cu, dogleg=63)
route("J_STAMP_17", 10, "J_EXEC", 1, "EXEC_N", pcbnew.B_Cu, dogleg=14)

route("J_STAMP_6", 26, "R1", 1, "I2S_BCLK_SRC", pcbnew.F_Cu, dogleg=17)
route("J_STAMP_6", 26, "R2", 1, "I2S_BCLK_SRC", pcbnew.F_Cu, dogleg=23)
route("R1", 2, "J_MIC", 4, "I2S_BCLK_MIC", pcbnew.F_Cu, dogleg=21)
route("R2", 2, "J_AMP_SIG", 2, "I2S_BCLK_AMP", pcbnew.F_Cu, dogleg=42)
route("J_STAMP_17", 5, "R3", 1, "I2S_WS_SRC", pcbnew.B_Cu, dogleg=29)
route("J_STAMP_17", 5, "R4", 1, "I2S_WS_SRC", pcbnew.B_Cu, dogleg=35)
route("R3", 2, "J_MIC", 5, "I2S_WS_MIC", pcbnew.F_Cu, dogleg=25)
route("R4", 2, "J_AMP_SIG", 1, "I2S_WS_AMP", pcbnew.F_Cu, dogleg=38)
route("J_STAMP_17", 7, "R5", 1, "I2S_DOUT_SRC", pcbnew.B_Cu, dogleg=43)
route("R5", 2, "J_AMP_SIG", 3, "I2S_DOUT_AMP", pcbnew.F_Cu, dogleg=46)
route("J_STAMP_17", 9, "J_MIC", 3, "I2S_DIN_MIC", pcbnew.B_Cu, dogleg=15)
route("J_STAMP_17", 9, "R7", 1, "I2S_DIN_MIC", pcbnew.B_Cu, dogleg=57)
route("J_STAMP_17", 1, "J_AMP_PWR", 1, "AMP_SD", pcbnew.B_Cu, dogleg=54)
route("J_STAMP_17", 1, "R6", 1, "AMP_SD", pcbnew.B_Cu, dogleg=51)

route("J_STAMP_17", 17, "J_SOLIST", 1, "I2C_SCL", pcbnew.F_Cu, dogleg=48)
route("J_STAMP_17", 17, "J_SERVO_CTRL", 2, "I2C_SCL", pcbnew.F_Cu, dogleg=70)
route("J_STAMP_17", 15, "J_SOLIST", 3, "I2C_SDA", pcbnew.B_Cu, dogleg=50)
route("J_STAMP_17", 15, "J_SERVO_CTRL", 3, "I2C_SDA", pcbnew.B_Cu, dogleg=72)

route("J_SOLIST", 6, "J_TFT_PWR", 1, "TFT_RST_N", pcbnew.B_Cu, dogleg=34)
route("J_SOLIST", 7, "J_TFT_SIG", 5, "TFT_DC", pcbnew.B_Cu, dogleg=32)
route("J_SOLIST", 9, "J_TFT_SIG", 4, "TFT_MOSI", pcbnew.B_Cu, dogleg=30)
route("J_SOLIST", 12, "J_TFT_SIG", 3, "TFT_SCK", pcbnew.F_Cu, dogleg=28)
route("J_SOLIST", 14, "J_TFT_PWR", 2, "TFT_CS_N", pcbnew.F_Cu, dogleg=26)
route("J_SOLIST", 14, "R9", 1, "TFT_CS_N", pcbnew.F_Cu, dogleg=25)
route("J_SOLIST", 10, "J_PCA_OE", 1, "PCA_OE", pcbnew.F_Cu, dogleg=67)
route("J_SOLIST", 10, "R8", 1, "PCA_OE", pcbnew.F_Cu, dogleg=65)

# Power branches use wide tracks; the separated servo rail never joins logic.
route("J_PWR_LOGIC", 1, "C1", 1, "+5V_LOGIC", pcbnew.F_Cu, 1.0)
route("J_PWR_LOGIC", 1, "C2", 1, "+5V_LOGIC", pcbnew.F_Cu, 1.0)
route("J_PWR_LOGIC", 1, "J_AMP_PWR", 3, "+5V_LOGIC", pcbnew.B_Cu, 1.0, 61)
route("J_PWR_LOGIC", 1, "J_TFT_PWR", 4, "+5V_LOGIC", pcbnew.B_Cu, 1.0, 12)
route("J_PWR_LOGIC", 1, "J_TFT_SIG", 2, "+5V_LOGIC", pcbnew.B_Cu, 1.0, 11)
route("J_PWR_LOGIC", 1, "JP_STAMP_5V", 1, "+5V_LOGIC", pcbnew.F_Cu, 1.0)
route("JP_STAMP_5V", 2, "J_STAMP_17", 13, "+5V_STAMP", pcbnew.F_Cu, 0.5, 60)
route("J_PWR_SERVO", 1, "C3", 1, "+5V_LOGIC", pcbnew.F_Cu, 1.5)
route("J_PWR_SERVO", 1, "J_SERVO_5V_OUT", 1, "+5V_LOGIC", pcbnew.F_Cu, 1.5, 78)

route("J_STAMP_6", 28, "J_MIC", 2, "+3V3_STAMP", pcbnew.B_Cu, 0.5, 13)
route("J_STAMP_6", 28, "J_SERVO_CTRL", 4, "+3V3_STAMP", pcbnew.B_Cu, 0.5, 73)
route("J_STAMP_6", 28, "R8", 2, "+3V3_STAMP", pcbnew.B_Cu, 0.5, 64)
route("J_STAMP_6", 28, "R9", 2, "+3V3_STAMP", pcbnew.B_Cu, 0.5, 24)
route("J_STAMP_6", 28, "C4", 1, "+3V3_STAMP", pcbnew.B_Cu, 0.5, 27)

# Board outline, labels, and manufacturing notes.
bottom = 82
for a, b in [((10, 10), (120, 10)), ((120, 10), (120, bottom)),
             ((120, bottom), (10, bottom)), ((10, bottom), (10, 10))]:
    edge = pcbnew.PCB_SHAPE(board)
    edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
    edge.SetStart(point(*a))
    edge.SetEnd(point(*b))
    edge.SetLayer(pcbnew.Edge_Cuts)
    edge.SetWidth(mm(0.1))
    board.Add(edge)

for x, y in [(14, 14), (116, 14), (14, bottom - 4), (116, bottom - 4)]:
    add_fp(f"H{len([f for f in board.GetFootprints() if f.GetReference().startswith('H')]) + 1}",
           "M3", "MountingHole", "MountingHole_3.2mm_M3", x, y)


def text_item(text: str, x: float, y: float, size: float = 1.0,
              layer: int = pcbnew.F_SilkS, rotation: float = 0) -> None:
    item = pcbnew.PCB_TEXT(board)
    item.SetText(text)
    item.SetPosition(point(x, y))
    item.SetLayer(layer)
    item.SetTextSize(pcbnew.VECTOR2I(mm(size), mm(size)))
    item.SetTextThickness(mm(0.15))
    item.SetTextAngle(pcbnew.EDA_ANGLE(rotation, pcbnew.DEGREES_T))
    board.Add(item)


text_item(f"IchiPing Solist-AI / Stamp-S3A Interposer Rev.B {VARIANT.upper()}", 61, bottom - 2, 1.1)
text_item("ALL HARNESS: GENUINE JST XH 2.50mm", 61, 11.5, 0.9)
text_item("MIC PIN2=3.3V (NOT UNO-Q 1.8V)", 100, 29, 0.8)
text_item("JP_STAMP_5V: REMOVE WHEN USB POWERED", 96, 59, 0.8)
text_item("SERVO 5V SEPARATE", 105, 79, 0.8)
text_item("SOLIST 2x7: PIN 1 MARK", 24, 66, 0.8)
text_item("STAMP TOP: ANT UP / USB DOWN", 61, 37, 0.8)
text_item("X-MIRRORED SOCKET VIEW", 61, 39, 0.8)
text_item("M1-1", 72, 11.8, 0.8)

# Electrical values remain readable after assembly.  Reference designators are
# provided by the footprints; these compact labels add the requested values.
value_labels = [
    ("R1 33R", 79, 20.1), ("R2 33R", 79, 26.1),
    ("R3 33R", 79, 32.1), ("R4 33R", 79, 38.1),
    ("R5 33R", 79, 46.1), ("R6 10k", 79, 54.1),
    ("R7 100k", 79, 60.1),
    ("R8 10k", 43 if VARIANT == "pcba" else 55,
     67.1 if VARIANT == "pcba" else 70.1),
    ("R9 10k", 30, 28.1),
    ("C1 470uF", 92, 55.0), ("C2 100nF", 70, 69.0),
    ("C4 100nF", 98, 31.0),
    ("C5 10uF", 98, 53.0),
]
for label, x, y in value_labels:
    text_item(label, x, y, 0.80)
text_item("C3 1000uF", 118.0, 64.0, 0.80, rotation=90)

# Stamp body projection.  Its antenna end intentionally protrudes about 1 mm
# beyond the carrier edge; keep the region above the first pads clear.
for a, b in [((52, 9), (70, 9)), ((70, 9), (70, 35)),
             ((70, 35), (52, 35)), ((52, 35), (52, 9))]:
    shape = pcbnew.PCB_SHAPE(board)
    shape.SetShape(pcbnew.SHAPE_T_SEGMENT)
    shape.SetStart(point(*a))
    shape.SetEnd(point(*b))
    shape.SetLayer(pcbnew.F_SilkS)
    shape.SetWidth(mm(0.2))
    board.Add(shape)
text_item("ANT", 61, 10.2, 0.8)
text_item("USB", 61, 34, 0.8)


def add_antenna_keepout(layer: int) -> None:
    keepout = pcbnew.ZONE(board)
    keepout.SetLayer(layer)
    keepout.SetIsRuleArea(True)
    keepout.SetDoNotAllowTracks(True)
    keepout.SetDoNotAllowVias(True)
    keepout.SetDoNotAllowZoneFills(True)
    outline = keepout.Outline()
    outline.NewOutline()
    for x, y in [(54.0, 10.5), (67.0, 10.5), (67.0, 18.0), (54.0, 18.0)]:
        outline.Append(point(x, y))
    board.Add(keepout)


add_antenna_keepout(pcbnew.F_Cu)
add_antenna_keepout(pcbnew.B_Cu)

# GND pours on both copper layers.  The concave top-edge notch excludes copper
# below the Stamp-S3A antenna while retaining copper beside its socket rows.
def add_ground_zone(layer: int) -> None:
    zone = pcbnew.ZONE(board)
    zone.SetLayer(layer)
    zone.SetNet(net("GND"))
    zone.SetLocalClearance(mm(0.30))
    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
    outline = zone.Outline()
    outline.NewOutline()
    for x, y in [
        (10.5, 10.5), (52.0, 10.5), (52.0, 18.0), (70.0, 18.0),
        (70.0, 10.5), (119.5, 10.5), (119.5, bottom - 0.5),
        (10.5, bottom - 0.5),
    ]:
        outline.Append(point(x, y))
    board.Add(zone)


if os.environ.get("PLACEMENT_ONLY") != "1":
    add_ground_zone(pcbnew.F_Cu)
    add_ground_zone(pcbnew.B_Cu)

# The reviewed routes above remain useful as deterministic escape-routing
# documentation.  PLACEMENT_ONLY=1 removes them so an external Specctra router
# can start from the same placement without inheriting crossing tracks.
if os.environ.get("PLACEMENT_ONLY") == "1":
    for track in list(board.GetTracks()):
        board.Remove(track)

pcbnew.SaveBoard(str(OUT), board)
print(f"Generated {OUT}")

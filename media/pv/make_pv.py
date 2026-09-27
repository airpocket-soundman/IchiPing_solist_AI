"""Build the IchiPing on Solist-AI PV (media/pv/IchiPing_on_Solist-AI_PV_v9.mp4).

    python media/figures/render_figures.py   # figures -> docs/protopedia/ (if changed)
    python media/pv/tts_narration.py         # narration -> media/pv/narration/*.mp3 (if changed)
    python media/pv/make_pv.py               # -> media/pv/build/IchiPing_on_Solist-AI_PV.mp4

Scenes are cut from media/footage, the figures in docs/protopedia and media/figures/external.
Each scene lasts as long as its narration (+ LEAD / TAIL); footage keeps its own sound
(switch clicks, servos, the PRBS chirp) at clip_gain, and the chirp is lifted by `boost`.
Conditions and the scene list are described in media/README.md.
"""
import os, subprocess, sys, wave
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FF = imageio_ffmpeg.get_ffmpeg_exe()
MEDIA = ROOT / "media"
PUB = ROOT / "docs" / "protopedia"
OUT = MEDIA / "pv" / "build"
TMP = OUT / "segments"
NAR_MP3 = MEDIA / "pv" / "narration"
NAR = OUT / "narration_wav"
TMP.mkdir(parents=True, exist_ok=True)
NAR.mkdir(parents=True, exist_ok=True)
FONT_B = r"C:/Windows/Fonts/NotoSansJP-Bold.ttf"
W, H, FPS = 1920, 1080, 30
LEAD, TAIL = 0.15, 0.3          # silence before / after narration inside a segment

def font(size):
    return ImageFont.truetype(FONT_B, size)

def nar_len(nid):
    wav = NAR / f"{nid}.wav"
    if not wav.exists():
        subprocess.run([FF, "-y", "-v", "error", "-i", str(NAR_MP3 / f"{nid}.mp3"), "-ar", "48000", "-ac", "2", str(wav)], check=True)
    w = wave.open(str(wav))
    return w.getnframes() / w.getframerate()

def caption_png(name, text, sub=None, band=False, top=False):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font(46)
    lines = text.split("\n")
    lh = 64
    h = lh * len(lines) + (46 if sub else 0) + 40
    y0 = 36 if top else H - h - (0 if band else 40)
    if band:
        d.rectangle([0, y0, W, H], fill=(10, 24, 44, 255))
    else:
        widths = [d.textlength(l, font=f) for l in lines]
        bw = max(widths + ([d.textlength(sub, font=font(30))] if sub else [])) + 80
        d.rounded_rectangle([(W - bw) / 2, y0, (W + bw) / 2, y0 + h], 18, fill=(10, 24, 44, 215))
    y = y0 + 20
    for l in lines:
        tw = d.textlength(l, font=f)
        d.text(((W - tw) / 2, y), l, font=f, fill=(255, 255, 255, 255))
        y += lh
    if sub:
        fs = font(30)
        tw = d.textlength(sub, font=fs)
        d.text(((W - tw) / 2, y + 2), sub, font=fs, fill=(170, 210, 255, 255))
    p = str(TMP / f"{name}.png")
    im.save(p)
    return p

def slide_png(name, src, bottom_band=170, bg=(255, 255, 255), crop=None, title=None):
    im = Image.open(src).convert("RGB")
    if crop:
        im = im.crop(crop)
    top = 0
    c = Image.new("RGB", (W, H), bg)
    if title:
        d = ImageDraw.Draw(c)
        d.text((60, 40), title, font=font(54), fill=(16, 42, 67))
        top = 130
    area_h = H - bottom_band - top
    s = min((W - 60) / im.width, area_h / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    c.paste(im, ((W - im.width) // 2, top + (area_h - im.height) // 2))
    p = str(TMP / f"{name}_slide.png")
    c.save(p)
    return p

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        print(" ".join(cmd)[:500]); print(r.stderr[-3000:]); sys.exit(1)

ENC = ["-color_range", "tv", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-r", str(FPS),
       "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]

def vfade(d):
    return f"fade=t=in:st=0:d=0.3,fade=t=out:st={d-0.3:.2f}:d=0.3"

def build(i, kind, src, dur_min, caps, nid=None, ss=0.0, speed=1.0, clip_gain=5.6, src_t=None, boost=None):
    """kind: 'img' or 'vid'. nid: narration id, or list of (id, start seconds).
    Clip audio keeps its level (boosted by clip_gain, no ducking) so chirps and switch clicks stay audible."""
    nars = [] if not nid else ([(nid, LEAD)] if isinstance(nid, str) else list(nid))
    end = max([st + nar_len(n) for n, st in nars], default=0.0)
    dur = round(max(dur_min, end + TAIL if nars else 0.0), 2)
    out = str(TMP / f"seg{i:02d}.mp4")
    inputs = []
    if kind == "img":
        inputs += ["-loop", "1", "-t", f"{dur}", "-i", src]
    else:
        inputs += ["-ss", f"{ss}", "-t", f"{(src_t if src_t else dur*speed):.3f}", "-i", src]
    for c, _, _ in caps:
        inputs += ["-loop", "1", "-t", f"{dur}", "-i", c]
    ai = len(caps) + 1
    for n, _ in nars:
        inputs += ["-i", str(NAR / f"{n}.wav")]
    if not nars:
        inputs += ["-f", "lavfi", "-t", f"{dur}", "-i", "anullsrc=r=48000:cl=stereo"]
    if kind == "img":
        chain = [f"[0:v]scale={W}:{H},setsar=1,fps={FPS},format=yuv420p[b0]"]
    else:
        chain = [f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,"
                 f"setpts=PTS/{speed},fps={FPS},tpad=stop_mode=clone:stop_duration=30,scale=out_range=tv,format=yuv420p[b0]"]
    last = "b0"
    for k, (_, t0, t1) in enumerate(caps):
        t1 = dur if t1 is None else t1
        chain.append(f"[{last}][{k+1}:v]overlay=0:0:enable='between(t,{t0},{t1})'[b{k+1}]")
        last = f"b{k+1}"
    chain.append(f"[{last}]{vfade(dur)},trim=duration={dur}[v]")
    labels = []
    if nars:
        for k, (n, st) in enumerate(nars):
            ms = int(st * 1000)
            chain.append(f"[{ai+k}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={ms}|{ms},volume=1.3,apad=whole_dur={dur}[n{k}]")
            labels.append(f"[n{k}]")
    else:
        chain.append(f"[{ai}:a]aformat=channel_layouts=stereo,apad=whole_dur={dur}[n0]")
        labels.append("[n0]")
    if kind == "vid":
        at = f"atempo={speed}," if speed != 1.0 else ""
        chain.append(f"[0:a]{at}aresample=48000,aformat=channel_layouts=stereo,volume={clip_gain}" + (f",volume={boost[2]}:enable='between(t,{boost[0]},{boost[1]})'" if boost else "") + f",apad=whole_dur={dur}[ca]")
        labels.append("[ca]")
    chain.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest,atrim=duration={dur},"
                 f"afade=t=in:st=0:d=0.15,afade=t=out:st={dur-0.3:.2f}:d=0.3[a]")
    run([FF, "-y", "-v", "error", *inputs, "-filter_complex", ";".join(chain),
         "-map", "[v]", "-map", "[a]", "-t", f"{dur}", *ENC, out])
    return out, dur

F = lambda n: str(PUB / n)
EXT = lambda n: str(MEDIA / "figures" / "external" / n)
VID = lambda n: str(MEDIA / "footage" / n)
segs = []
total = 0.0
def add(*a, **k):
    global total
    p, d = build(len(segs), *a, **k)
    segs.append(p); total += d
    print(f"seg{len(segs)-1:02d} {d:5.2f}s  total {total:6.2f}")

add("img", F("fig_hero.jpg"), 5.0, [(caption_png("c_title", "ROHM EDGE HACK CHALLENGE 2026 エントリー作品", top=True), 0.5, None)], nid="n01")
add("img", slide_png("manga", EXT("manga.png"), 0, (0, 0, 0)), 3.8,
    [(caption_png("c_prob", "外出先で雨。「あれ、窓閉めたっけ?」"), 0.3, None)], nid="n02")
add("vid", VID("VID_20260927_131743.mp4"), 7.0,
    [(caption_png("c_concept", "1 個のマイクと 1 発の Ping で\n3 部屋の窓 3 枚・扉 2 枚 = 32 通りの開閉を当てる", top=True), 0.3, None)], nid="n03", ss=0.0)
add("img", slide_png("house", EXT("house.png")), 5.0,
    [(caption_png("c_house", "Room A にスピーカとマイク。扉が閉じると、奥の部屋は本来「聞こえない」", band=True), 0, 5.6),
     (caption_png("c_house2", "それでも、閉ざされた扉の奥のわずかな反響音の特徴まで学習できた", band=True), 5.6, None)], nid="n04")
add("vid", VID("VID_20260927_131620.mp4"), 6.0,
    [(caption_png("c_mech", "窓と扉はサーボで開閉。トグルの状態 = 模型の状態 = 正解ラベル", top=True), 0.3, None)], nid="n05", ss=0.0)
add("vid", VID("VID_20260927_131829.mp4"), 23.0, [
    (caption_png("c_op1", "① トグルで窓・扉を指定", "操作方法", top=True), 0.3, 6.0),
    (caption_png("c_op2", "② EXEC を押すと Ping を 2 秒鳴らして録音", "Listening...", top=True), 6.0, 10.2),
    (caption_png("c_op3", "③ Solist-AI が推論", "Inferring... のゲージが進む", top=True), 10.2, 19.5),
    (caption_png("c_op4", "④ inf(推論)と act(正解)が一致 → Complete Success", top=True), 19.5, None)],
    nid=[("n07a", 0.3), ("n07b", 10.2)], ss=0.8, src_t=20.3, boost=(7.4, 10.0, 8.0))
add("vid", VID("VID_20260927_131620.mp4"), 5.5, [
    (caption_png("c_d1", "全部開けた状態(11111)も正解", "2 倍速", top=True), 0.3, None)], nid="n09", ss=19.0, speed=2.0)
add("img", slide_png("fftdiff", EXT("fftdiff_s00000_vs_s10000.png")), 5.0,
    [(caption_png("c_fft", "全閉との差分を取ると、スピーカや部屋の癖が消え「変わった分」だけが残る", band=True), 0, None)], nid="n11")
add("img", slide_png("system", F("fig_system.png")), 5.0, [
    (caption_png("c_sys1", "コア処理はすべて Solist-AI:特徴抽出・推論・学び直し・表示・サーボ制御", band=True), 0, 7.5),
    (caption_png("c_sys2", "Stamp-S3A は評価ボードに無い I²S と、足りない GPIO を補うだけ", band=True), 7.5, None)], nid="n13")
add("img", slide_png("pipe", F("fig_pipeline.png"), crop=(0, 0, 1600, 430)), 5.0,
    [(caption_png("c_ai1", "差分スペクトル → int8 CNN 前段(Cortex-M0+)→ ELM(AxlCORE)→ 32 クラス", band=True), 0, 5.0),
     (caption_png("c_ai2", "CNN は int8 で積算し、層ごとの再量子化だけ float32 で計算して精度を保つ", band=True), 5.0, None)], nid="n15")
add("img", slide_png("params", F("fig_params.png")), 5.0, [
    (caption_png("c_par1", "ELM だけだと、学習外のセッションでの 32 クラス分類の正解率は 66%", band=True), 0, 8.0),
    (caption_png("c_par2", "CNN 前段を足して汎化させても、パラメータ数は PC の理想モデルの約 1/7", band=True), 8.0, None)], nid="n16")
add("img", slide_png("variation", F("fig_variation.png")), 5.0, [
    (caption_png("c_var1", "反響音は気温や背景音で変わり、収録セッションごとにデータがばらつく", band=True), 0, 6.0),
    (caption_png("c_var2", "① 幅広いデータで事前学習 ② 大きな変化は ODL でその場で補う", band=True), 6.0, None)], nid="n18a")
add("img", slide_png("auto", F("fig_autocollect.png")), 5.0, [
    (caption_png("c_auto1", "模型が自分で窓と扉を動かすので、データ採取は無人", band=True), 0, 5.5),
    (caption_png("c_auto2", "条件の違う 13 セッション・4,290 フレームを自動で採取・採点", band=True), 5.5, None)], nid="n18")
add("img", slide_png("acc", F("fig_accuracy.png")), 5.0, [
    (caption_png("c_acc1", "汎化性能を上げ、観測しやすい 14 クラスだけでなく 32 クラスまで", band=True), 0, 5.0),
    (caption_png("c_acc2", "閉じた扉の向こうも含む 32 クラスの推論に、実機で 100% 成功", band=True), 5.0, None)], nid="n19")
add("img", slide_png("odl3", F("fig_odl.png")), 5.0, [
    (caption_png("c_odl1", "Solist-AI の目玉機能:オンデバイス学習(ODL)", band=True), 0, 8.0),
    (caption_png("c_odl2", "EXEC 長押し → 全 32 クラスの開閉を巡回 → Solist-AI がその場で β を更新", band=True), 8.0, 15.5),
    (caption_png("c_odl3", "PC もクラウドも不要。事前学習に含めていない条件でも 89.4% → 100%", band=True), 15.5, None)], nid="n17")
end_png = str(TMP / "end.png")
im = Image.new("RGB", (W, H), (8, 20, 40)); d = ImageDraw.Draw(im)
fr = Image.open(MEDIA / "figures" / "frames" / "f829_42.8.jpg").convert("RGB").resize((1152, 648))
im.paste(fr, (384, 90))
t = "IchiPing on Solist-AI"; f1 = font(84); d.text(((W - d.textlength(t, font=f1)) / 2, 770), t, font=f1, fill=(255, 255, 255))
t2 = "github.com/airpocket-soundman/IchiPing_solist_AI"; f2 = font(36); d.text(((W - d.textlength(t2, font=f2)) / 2, 900), t2, font=f2, fill=(160, 205, 255))
im.save(end_png)
add("img", end_png, 5.0, [], nid="n20")

lst = TMP / "list.txt"
with open(lst, "w", encoding="utf-8") as fh:
    for p in segs:
        fh.write(f"file '{Path(p).as_posix()}'\n")
joined = OUT / "joined.mp4"
run([FF, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)])
# final pass: limiter so the boosted footage never clips
final = OUT / "IchiPing_on_Solist-AI_PV.mp4"
run([FF, "-y", "-v", "error", "-i", str(joined), "-c:v", "copy", "-af", "alimiter=limit=0.75:level=false",
     "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(final)])
print("done", final, round(total, 2))

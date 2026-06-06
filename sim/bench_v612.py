"""IchiPing標準プロトコルでの AE-bank ベンチ (14cls/32cls) + Flash 試算。

プロトコル (IchiPing 実装準拠):
  - 学習データ: v6-v12 (7 run)
  - cross-baseline: 複数 run の s00000 を baseline にして diff を多重生成 (baseline jitter)
  - strong aug (特徴空間): SpectralJitter σ0.6dB, FreqMask、+ 音声 GaussianHiss(任意)
  - 評価: eval_quiet/noise_low/noise_high を、単一 baseline(eval_quiet)で diff (=1回校正→別環境運用)
  - 方式: one-class detector(クラス別オートエンコーダ)を 14/32 個 → argmin 再構成誤差
特徴: FFT(audio − baseline)、NFFT=1024→512bin、帯域クロップ(ネイティブ分解能)。
データ形式は実機 bfloat16 だが sim は float (int8 でも 100% 確認済なので bf16 は余裕)。
"""
from __future__ import annotations

import sys, wave
from pathlib import Path
import numpy as np

from common import CAPTURES_ROOT, CACHE_DIR, parse_state_label, class_idx_14
from solist_elm import _act, accuracy

NFFT = 1024; HOP = 512; BIN_HZ = 16000.0 / NFFT
TRAIN_RUNS = [f"full_32_train_v{i}" for i in (6, 7, 8, 9, 10, 11, 12)]
BASELINE_RUNS = ["full_32_train_v6", "full_32_train_v9", "full_32_train_v12"]  # cross-baseline jitter
EVAL_SETS = ["eval_quiet", "eval_noise_low", "eval_noise_high"]
EVAL_BASELINE = "eval_quiet"   # 1回校正の基準 (cross-baseline運用)
_WIN = np.hanning(NFFT)


def load_wav(p):
    with wave.open(str(p), "rb") as w: raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


def baseline_of(run):
    bw = sorted((CAPTURES_ROOT / run / "s00000").glob("frame_*.wav"))
    L = len(load_wav(bw[0]))
    return np.mean(np.stack([load_wav(p)[:L] for p in bw]), axis=0)


def diff_spec(audio, baseline):
    n = min(len(audio), len(baseline)); d = audio[:n] - baseline[:n]
    starts = range(0, n - NFFT + 1, HOP)
    seg = np.stack([d[s:s+NFFT] for s in starts]) * _WIN      # (nseg, NFFT)
    mag = np.abs(np.fft.rfft(seg, axis=1)).mean(0)[1:]        # (512,)
    return np.maximum(20*np.log10(mag + 1e-9), -80.0).astype(np.float32)


def build_run(run, baseline, tag):
    """run の全frameを baseline で diff。npzキャッシュ。returns X,y14,y32"""
    CACHE_DIR.mkdir(exist_ok=True)
    out = CACHE_DIR / f"v612_{run}__{tag}.npz"
    if out.exists():
        z = np.load(out); return z["X"], z["y14"], z["y32"]
    X, y14, y32 = [], [], []
    for sd in sorted((CAPTURES_ROOT / run).iterdir()):
        bits = parse_state_label(sd.name)
        if bits is None: continue
        c14 = class_idx_14(bits); c32 = int(sum(int(b)<<k for k,b in enumerate(bits)))
        for wav in sorted(sd.glob("frame_*.wav")):
            X.append(diff_spec(load_wav(wav), baseline)); y14.append(c14); y32.append(c32)
    X = np.stack(X); y14 = np.array(y14); y32 = np.array(y32)
    np.savez_compressed(out, X=X, y14=y14, y32=y32)
    return X, y14, y32


def crop(X, lo_hz, hi_hz):
    lo = max(0, int(round(lo_hz/BIN_HZ))-1); hi = min(X.shape[1], int(round(hi_hz/BIN_HZ)))
    return X[:, lo:hi], hi-lo


class AEBank:
    def __init__(self, K, ridge, act="hard_sigmoid", seed=0):
        self.K=K; self.ridge=ridge; self.act=act; self.seed=seed
    def fit(self, X, y, classes):
        rng=np.random.default_rng(self.seed); D=X.shape[1]
        self.alpha=rng.standard_normal((D,self.K))/np.sqrt(D); self.bias=rng.standard_normal(self.K)*0.1
        H=_act(self.act, X@self.alpha+self.bias); self.betas={}
        for c in classes:
            Hc=H[y==c]; Xc=X[y==c]
            self.betas[c]=np.linalg.solve(Hc.T@Hc+self.ridge*np.eye(self.K), Hc.T@Xc)
        self.classes=list(classes); return self
    def scores(self, X):
        H=_act(self.act, X@self.alpha+self.bias)
        return np.stack([((X-H@self.betas[c])**2).mean(1) for c in self.classes],1)


def vote(S, y, y32, classes, k=0):
    classes=np.array(classes); cor=tot=0
    for s in np.unique(y32):
        idx=np.where(y32==s)[0]
        groups=[idx] if k<=0 else [idx[i:i+k] for i in range(0,len(idx),k)]
        for g in groups:
            cor+=int(classes[S[g].sum(0).argmin()]==y[g][0]); tot+=1
    return cor/tot


def collapse14(y32arr):
    return np.array([class_idx_14(np.array([(v>>k)&1 for k in range(5)])) for v in y32arr])


def specaug(X, y, n_extra=1, sigma=0.6, seed=0):
    """SpectralJitter で学習データを n_extra 倍に増やす (strong aug 特徴空間)。"""
    rng=np.random.default_rng(seed); Xs=[X]; ys=[y]
    for _ in range(n_extra):
        Xs.append(X + rng.normal(0, sigma, X.shape).astype(X.dtype)); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def build_run_noisy(run, baseline, seed, snr_db=(5, 25)):
    """GaussianHiss を音声に加えてから diff (strong aug 音声空間, ノイズ頑健化)。キャッシュ不可(乱数)。"""
    rng = np.random.default_rng(seed)
    X, y14, y32 = [], [], []
    for sd in sorted((CAPTURES_ROOT / run).iterdir()):
        bits = parse_state_label(sd.name)
        if bits is None: continue
        c14 = class_idx_14(bits); c32 = int(sum(int(b)<<k for k,b in enumerate(bits)))
        for wav in sorted(sd.glob("frame_*.wav")):
            a = load_wav(wav)
            snr = rng.uniform(*snr_db); rms = np.sqrt((a**2).mean())+1e-12
            a = a + rng.normal(0, rms/10**(snr/20), a.shape)
            X.append(diff_spec(a, baseline)); y14.append(c14); y32.append(c32)
    return np.stack(X), np.array(y14), np.array(y32)


def main():
    # --- 学習特徴: v6-v12 を cross-baseline (3 baseline) で多重生成 ---
    print("[build] train features (v6-v12 × 3 baselines, cross-baseline)...", flush=True)
    bls = {r: baseline_of(r) for r in BASELINE_RUNS}
    Xtr=[]; y14tr=[]; y32tr=[]
    for run in TRAIN_RUNS:
        for br, bl in bls.items():
            X,y14,y32 = build_run(run, bl, f"bl_{br[-3:]}")
            Xtr.append(X); y14tr.append(y14); y32tr.append(y32)
    Xtr=np.concatenate(Xtr); y14tr=np.concatenate(y14tr); y32tr=np.concatenate(y32tr)
    print(f"  train X={Xtr.shape} (cross-baseline applied)")

    # --- 評価特徴: eval_* を単一 baseline(eval_quiet) で diff (1回校正→別環境) ---
    print("[build] eval features (baseline=eval_quiet, cross-baseline運用)...", flush=True)
    eval_bl = baseline_of(EVAL_BASELINE)
    evals={}
    for es in EVAL_SETS:
        evals[es] = build_run(es, eval_bl, "bl_evalquiet")

    # --- strong aug 音声空間: GaussianHiss ノイズ付与 diff を追加 (v9 baseline) ---
    print("[build] audio-noise aug (GaussianHiss SNR 5-25dB)...", flush=True)
    Xn=[]; y14n=[]; y32n=[]
    for i,run in enumerate(TRAIN_RUNS):
        X,y14,y32 = build_run_noisy(run, bls["full_32_train_v9"], seed=100+i)
        Xn.append(X); y14n.append(y14); y32n.append(y32)
    Xn=np.concatenate(Xn); y14n=np.concatenate(y14n); y32n=np.concatenate(y32n)

    LO, HI = 400, 3000
    Xtr_c, D = crop(Xtr, LO, HI)
    Xn_c, _ = crop(Xn, LO, HI)
    # cross-baseline clean + SpectralJitter + 音声ノイズaug を結合
    Xj, y14j = specaug(Xtr_c, y14tr, n_extra=1)
    _,  y32j = specaug(Xtr_c, y32tr, n_extra=1)
    Xtr_a   = np.concatenate([Xj, Xn_c]);  y14tr_a = np.concatenate([y14j, y14n])
    y32tr_a = np.concatenate([y32j, y32n])
    print(f"  total train(after aug) X={Xtr_a.shape}")

    for ncls, ytag, ytr, yall in ((14,"14cls",y14tr_a,y14tr),(32,"32cls",y32tr_a,y32tr)):
        classes=list(range(ncls))
        for K in (16, 24, 32):
            bank=AEBank(K=K, ridge=1e-1).fit(Xtr_a, ytr, classes)
            print(f"\n=== {ytag} AE-bank D={D} K={K} (hard_sigmoid, cross-baseline, strong-aug) ===")
            for es in EVAL_SETS:
                Xe,y14e,y32e = evals[es]; Xec,_=crop(Xe,LO,HI)
                S=bank.scores(Xec)
                yt = y14e if ncls==14 else y32e
                a_f=accuracy(yt, np.array(classes)[S.argmin(1)])
                a_v=vote(S, yt, y32e, classes, 0)
                if ncls==32:
                    p14=collapse14(np.array(classes)[S.argmin(1)]); t14=collapse14(y32e)
                    extra=f"  →14collapse frame={accuracy(t14,p14):.1%}"
                else: extra=""
                print(f"  {es:16}: frame={a_f:6.1%}  vote-all={a_v:6.1%}{extra}")
            # Flash 試算 (bfloat16: 2byte) — β のみ保存(推論用), α=seed再生成
            beta_bytes = D*K*2*ncls
            print(f"  Flash(β bank, bf16) = {beta_bytes/1024:.1f}KB  "
                  f"(+baseline 2s={32000*2/1024:.0f}KB) = {(beta_bytes+64000)/1024:.1f}KB / 256KB")


if __name__ == "__main__":
    main()

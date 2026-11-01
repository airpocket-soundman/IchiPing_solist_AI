"""Inspect a suspect held-out state without modifying recordings."""
import json, hashlib, wave
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eval_best_model import run_db, feats

root=Path(__file__).resolve().parent.parent
runs=[f'stamp_20261031_holdout_eval_20261031_s{i}_wav' for i in (1,2)]
data={r:run_db(r) for r in runs}
label=lambda c:'h'+''.join(str((int(c)>>k)&1) for k in (2,4,1,3,0))
target=10; other=14
summary={}
for r in runs:
    files=sorted((root/'captures'/r/'s01010').glob('*.wav'))
    stats=[]
    for file in files:
        with wave.open(str(file)) as w:
            rate=w.getframerate(); x=np.frombuffer(w.readframes(w.getnframes()),'<i2').astype(float)
        centered=x-x.mean()
        blocks=np.sqrt(np.mean(centered.reshape(-1,160)**2,axis=1))
        stats.append(dict(file=file.name,n=len(x),rate=rate,rms=float(np.sqrt(np.mean(centered**2))),mean=float(x.mean()),peak=float(np.max(np.abs(x))),clipped=int(np.sum((x>=32767)|(x<=-32768))),zero_fraction=float(np.mean(x==0)),block_rms_min=float(blocks.min()),block_rms_max=float(blocks.max()),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    summary[r]=dict(waveforms=stats)
# Compare session-1 target with every session-2 state using both dB and normalized spectra.
for representation in ('db','normalized'):
    a=data[runs[0]]; b=data[runs[1]]
    A=a['db']-a['base'] if representation=='db' else feats(a,a['base']).astype(float)
    B=b['db']-b['base'] if representation=='db' else feats(b,b['base']).astype(float)
    rankings=[]
    for c in (target,other):
        mean=A[a['y']==c].mean(0)
        distances=[(label(k),float(np.sqrt(np.mean((mean[12:346]-B[b['y']==k].mean(0)[12:346])**2)))) for k in range(32)]
        rankings.append(dict(source=label(c),nearest=sorted(distances,key=lambda z:z[1])[:8]))
    summary[representation+'_nearest_session2']=rankings
fig,axes=plt.subplots(2,1,figsize=(11,7))
hz=np.arange(1,1025)*16000/2048
for r in runs:
    d=data[r]
    for c in (target,other):
        v=d['db'][d['y']==c]-d['base']
        mean=v.mean(0); std=v.std(0)
        axes[0].plot(hz,mean,label=f"session {runs.index(r)+1} {label(c)}")
        axes[0].fill_between(hz,mean-std,mean+std,alpha=.12)
    files=sorted((root/'captures'/r/'s01010').glob('*.wav'))
    with wave.open(str(files[0])) as w:x=np.frombuffer(w.readframes(w.getnframes()),'<i2')
    axes[1].plot(np.arange(len(x))/16000,x,alpha=.5,label=f"session {runs.index(r)+1} target frame 0")
axes[0].set_xlim(100,2700);axes[0].set_ylabel('dB relative to session baseline');axes[0].legend()
axes[1].set_xlabel('seconds');axes[1].set_ylabel('PCM');axes[1].legend()
fig.tight_layout();fig.savefig(root/'docs/results/suspect_eval_waveforms_20261031.png',dpi=150)
(root/'docs/results/suspect_eval_waveforms_20261031.json').write_text(json.dumps(summary,indent=2))
for r in runs:
    s=summary[r]['waveforms'];print(r,'RMS',min(v['rms'] for v in s),max(v['rms'] for v in s),'clip',sum(v['clipped'] for v in s),'unique',len(set(v['sha256'] for v in s)))
for k in ('db_nearest_session2','normalized_nearest_session2'):print(k,summary[k])

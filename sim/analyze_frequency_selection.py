"""Training-only frequency screening; no held-out evaluation data used."""
import json
from pathlib import Path
import numpy as np
from eval_best_model import run_db,feats

root=Path(__file__).resolve().parent.parent
cfg=json.loads((root/'sim_export/solist_ds/candidates_20261031/stamp_shift0.01_base4_noise0/complete.json').read_text())
runs=cfg['sessions']; hz=np.arange(1,1025)*16000/2048
means=[]; variances=[]
for r in runs:
 d=run_db(r); X=feats(d,d['base']).astype(float)
 means.append(np.stack([X[d['y']==c].mean(0) for c in range(32)]))
 variances.append(np.stack([X[d['y']==c].var(0) for c in range(32)]))
M=np.stack(means); V=np.stack(variances)
between=M.mean(0).var(0)
drift=M.var(0).mean(0)
within=V.mean((0,1))
score=between/(drift+within+1e-8)
# Stability of C-pair differences across sessions; retain sign consistency.
pair_scores=[]
for c in range(32):
 if c&4:continue
 delta=M[:,c|4]-M[:,c]
 pair_scores.append(delta.mean(0)**2/(delta.var(0)+V[:,c].mean(0)+V[:,c|4].mean(0)+1e-8))
cscore=np.mean(pair_scores,axis=0)
valid=(hz>=100)&(hz<=6000)
rank=np.argsort(-np.where(valid,score,-1))[:334]
combined=score/(score.max()+1e-9)+cscore/(cscore.max()+1e-9)
crank=np.argsort(-np.where(valid,combined,-1))[:334]
bands=[]
for lo,hi in [(100,400),(400,1000),(1000,2000),(2000,2700),(2700,4000),(4000,6000),(6000,8000)]:
 mask=(hz>=lo)&(hz<hi)
 bands.append(dict(lo=lo,hi=hi,mean_score=float(score[mask].mean()),mean_c_score=float(cscore[mask].mean()),selected=int(np.sum(mask[rank])),selected_c=int(np.sum(mask[crank]))))
# Practical contiguous blocks totaling 334 bins; candidates only, not trained.
def block(lo,n):return list(range(lo,lo+n))
candidates={
 'baseline_low_contiguous':block(12,334),
 'ranked_334':sorted(rank.tolist()),
 'ranked_c_aware_334':sorted(crank.tolist()),
 'three_bands_334':block(12,64)+block(128,142)+block(384,128),
}
out=dict(method='Training-only ranking, between-class variance / (session drift + within-state variance); not CNN accuracy',sessions=runs,bands=bands,
 top_frequencies=[dict(hz=float(hz[k]),score=float(score[k]),c_score=float(cscore[k])) for k in np.argsort(-np.where(valid,score,-1))[:30]],candidates=candidates,
 caveat='Discontinuous bins require explicit CNN band-boundary handling and retraining; ranking does not prove accuracy improvement')
(root/'docs/results/frequency_selection_20261031.json').write_text(json.dumps(out,indent=2))
for b in bands:print(b,flush=True)
print('TOP',[(round(hz[k],1),round(score[k],2)) for k in np.argsort(-np.where(valid,score,-1))[:12]],flush=True)

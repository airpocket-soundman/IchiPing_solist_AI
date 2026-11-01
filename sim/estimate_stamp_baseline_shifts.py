"""Estimate relative spectral dilation from one baseline WAV per training run."""
import json
from pathlib import Path
import numpy as np
import estimate_run_shift as estimator

root=Path(__file__).resolve().parent.parent
cfg=json.loads((root/'sim_export/solist_ds/candidates_20261031/stamp_shift0.01_base4_noise0/complete.json').read_text())
runs=cfg['sessions']+['stamp_20261031_current_train_20261031_s1_wav','stamp_20261031_current_train_20261031_s2_wav','stamp_20261031_hidden_c_targeted_20261031_s1_wav']
spectra={}; rows=[]
for name in runs:
 d=root/'captures'/name/'s00000'
 wavs=estimator.baseline_wavs(d.parent,d)
 assert wavs,name
 wav=wavs[0]
 spectra[name]=10*np.log10(estimator.power(estimator.load_wav(wav))+1e-20)
 times=estimator.frame_times(d)
 rows.append(dict(session=name,wav=wav.name,time=times.get(wav.name)))
reference=runs[0]
for row in rows:
 name=row['session'];row['bands']={}
 for lo,hi in ((100,2700),(400,3000),(1000,6000)):
  estimator.LO=int(lo/estimator.BIN_HZ);estimator.HI=int(hi/estimator.BIN_HZ)
  shift,corr=estimator.fit_eps(spectra[reference],spectra[name])
  row['bands'][f'{lo}_{hi}']=dict(shift_percent=shift*100,correlation=corr,temperature_equivalent_degC_at20=shift/.0017)
 print(name,[(k,round(v['shift_percent'],3),round(v['correlation'],3)) for k,v in row['bands'].items()],flush=True)
out=dict(reference=reference,method='One baseline frame per session; Welch FFT8192, power smoothing ~31Hz; relative dilation fit +/-6%; temperature equivalence is illustrative, not measured temperature',rows=rows)
(root/'docs/results/stamp_baseline_shifts_20261031.json').write_text(json.dumps(out,indent=2))

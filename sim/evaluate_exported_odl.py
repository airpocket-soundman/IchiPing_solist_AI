"""Compare factory and exported device ODL beta on identical held-out audio."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from eval_full_data import build_run,norm_diff
from eval_best_model import run_db
from emit_frontend_model import int_forward,elm_input,mcu_reference
from eval_stamp_session import class14

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--odl',required=True)
    parser.add_argument('--sessions',nargs='+',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    model=Path(args.model);z=np.load(model);state=np.load(args.odl)
    if str(state['model_sha256'])!=hashlib.sha256(model.read_bytes()).hexdigest():raise RuntimeError('ODL model mismatch')
    train_config=json.loads((model.parent/'complete.json').read_text())
    if set(args.sessions).intersection(train_config['sessions']):raise RuntimeError('Evaluation overlaps training')
    qm=[dict(wq=z[f'conv{i}_wq'],M=z[f'conv{i}_M'],B=z[f'conv{i}_B'],k=int(z[f'conv{i}_k']),s=int(z[f'conv{i}_s'])) for i in range(3)]
    fc=dict(wq=z['fc_wq'],M=z['fc_M'],B=z['fc_B'])
    lo,hi=int(z['bin_lo']),int(z['bin_hi'])
    report=dict(model=str(model.resolve()),odl=str(Path(args.odl).resolve()),model_sha256=str(state['model_sha256']),modes={})
    for mode in ('session_baseline','device_odl_baseline'):
        rows={};truth=[];before=[];after=[]
        for session in args.sessions:
            d=build_run(session)
            if mode=='session_baseline':X=d['X'].astype(np.float32)
            else:X=norm_diff(run_db(session)['db'],state['baseline_q'].astype(np.float32)/256).astype(np.float16).astype(np.float32)
            y=d['y']
            q=np.clip(np.round(((X[:,lo:hi]-z['in_mu'])/z['in_sd'])/float(z['s_in'])),-127,127).astype(np.int8)
            E=int_forward(q,qm,fc);Z=elm_input(E,z['emb_mul'],z['emb_add'])
            bf=mcu_reference(Z,z['beta']).argmax(1);af=mcu_reference(Z,state['beta']).argmax(1)
            def metrics(pred):
                return dict(n=len(y),correct32=int((pred==y).sum()),accuracy32=float((pred==y).mean()),
                            correct14=sum(class14(int(a))==class14(int(b)) for a,b in zip(pred,y)))
            rows[session]=dict(factory=metrics(bf),odl=metrics(af),fixed=int(((bf!=y)&(af==y)).sum()),regressed=int(((bf==y)&(af!=y)).sum()))
            truth.extend(y.tolist());before.extend(bf.tolist());after.extend(af.tolist())
        y=np.array(truth)
        summary={}
        for label,pred in [('factory',np.array(before)),('odl',np.array(after))]:
            cm=np.zeros((32,32),int)
            for a,b in zip(y,pred):cm[a,b]+=1
            correct14=sum(class14(int(a))==class14(int(b)) for a,b in zip(y,pred))
            summary[label]=dict(n=len(y),correct32=int((pred==y).sum()),accuracy32=float((pred==y).mean()),
                                correct14=correct14,accuracy14=correct14/len(y),confusion=cm.tolist())
        report['modes'][mode]=dict(sessions=rows,summary=summary)
        print(mode,summary['factory']['accuracy32'],'->',summary['odl']['accuracy32'],flush=True)
    Path(args.out).write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()

"""Optional training-only balancing, hidden-C margin, and group validation hooks."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import eval_full_data as efd
import emit_frontend_model as em
import eval_best_model as eb

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--balance-session-class', action='store_true')
parser.add_argument('--c-pair-weight', type=float, default=0)
parser.add_argument('--robust-validation', action='store_true')
parser.add_argument('--forced-seed', type=int)
options, rest = parser.parse_known_args()
sys.argv = [sys.argv[0]] + rest
value = lambda key: rest[rest.index(key)+1]
start = rest.index('--val-sessions')+1
validation = []
for arg in rest[start:]:
    if arg.startswith('--'): break
    validation.append(arg)
groups = np.concatenate([np.full(len(efd.build_run(s)['y']), i) for i,s in enumerate(validation)])
weights_by_length = {}
scores = []


def build_train(runs, xbase, rng, stamp_rep=1):
    X, y, groups_train = [], [], []
    for index, run in enumerate(runs):
        d = eb.run_db(run)
        pool = runs if eb.BASELINE_REFERENCE_RUNS is None else [s for s in eb.BASELINE_REFERENCE_RUNS if s in runs]
        others = [s for s in pool if s != run and eb.family(s) == eb.family(run)]
        selected = rng.choice(others, size=min(xbase,len(others)), replace=False) if others else []
        for base in [d['base']]+[eb.run_db(str(s))['base'] for s in selected]:
            for _ in range(stamp_rep if eb.family(run)=='stamp' else 1):
                X.append(eb.feats(d,base)); y.append(d['y']); groups_train.append(np.full(len(d['y']),index))
    X, y, g = np.concatenate(X), np.concatenate(y), np.concatenate(groups_train)
    weights = np.ones(len(y), np.float32)
    if options.balance_session_class:
        # Equal mass per class, then equal mass per session containing that class.
        for c in np.unique(y):
            mask = y == c
            present = np.unique(g[mask])
            for session in present:
                cell = mask & (g == session)
                weights[cell] = 1/(len(present)*cell.sum())
        weights /= weights.mean()
    weights_by_length[len(y)] = weights
    return X, y


def elm_fit(Zp, y, lam):
    H = em.hard_sigmoid(Zp @ em.ALPHA)
    Y = np.eye(em.C)[y]
    w = weights_by_length.get(len(y), np.ones(len(y)))
    return np.linalg.solve(H.T @ (H*w[:,None])+lam*np.eye(em.ELM_HIDDEN), H.T @ (Y*w[:,None])).astype(np.float32)


def train_net(spec, fc, Xtr, ytr, Xva, yva, seed, epochs=200, patience=20):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    Xt=torch.as_tensor(Xtr.astype(np.float32),device=efd.DEV)
    yt=torch.as_tensor(ytr,device=efd.DEV)
    wt=torch.as_tensor(weights_by_length[len(ytr)],device=efd.DEV)
    band=Xt[:,efd.BLO:efd.BHI]
    if options.balance_session_class:
        mu=(band*wt[:,None]).sum(0)/wt.sum()
        sd=(((band-mu).square()*wt[:,None]).sum(0)/wt.sum()).sqrt()+1e-6
    else:
        mu,sd=band.mean(0),band.std(0)+1e-6
    prep=lambda A: ((A[:,efd.BLO:efd.BHI]-mu)/sd)[:,None,:]
    Xv=prep(torch.as_tensor(Xva.astype(np.float32),device=efd.DEV))
    yv=torch.as_tensor(yva,device=efd.DEV)
    vg=torch.as_tensor(groups,device=efd.DEV)
    net=efd.Net(spec,fc).to(efd.DEV)
    opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=6)
    gen=torch.Generator(device='cpu').manual_seed(seed)
    best=None; best_loss=float('inf'); stale=0
    for epoch in range(epochs):
        net.train()
        perm=torch.randperm(len(Xt),generator=gen).to(efd.DEV)
        for i in range(0,len(perm),512):
            b=perm[i:i+512]
            eps=(torch.rand(len(b),device=efd.DEV)*2-1)*efd.SHIFT_MAX
            logits=net(prep(efd.shift_batch(Xt[b],eps)))
            loss=F.cross_entropy(logits,yt[b],label_smoothing=.05,reduction='none')
            if options.c_pair_weight:
                # Bit 2 is C; penalize confusing a state with its C-flipped partner.
                correct=logits.gather(1,yt[b,None]).squeeze(1)
                partner=logits.gather(1,(yt[b]^4)[:,None]).squeeze(1)
                loss=loss+options.c_pair_weight*F.softplus(partner-correct+.5)
            loss=(loss*wt[b]).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            losses=F.cross_entropy(net(Xv),yv,reduction='none')
            per=torch.stack([losses[vg==g].mean() for g in range(len(validation))])
            vl=float((per.max()+per.mean())/2 if options.robust_validation else losses.mean())
        sched.step(vl)
        if vl < best_loss-1e-5:
            best_loss=vl; stale=0
            best={k:v.detach().clone() for k,v in net.state_dict().items()}
        else:
            stale+=1
            if stale>=patience: break
    net.load_state_dict(best); net.eval()
    net.in_mu=mu.cpu().numpy(); net.in_sd=sd.cpu().numpy()
    def apply(A,fn):
        with torch.no_grad():
            return np.concatenate([fn(prep(torch.as_tensor(A[i:i+4096].astype(np.float32),device=efd.DEV))).cpu().numpy() for i in range(0,len(A),4096)]).astype(np.float64)
    return net, lambda A:apply(A,net.front), lambda A:apply(A,net)


original_candidate=em.build_candidate
def build_candidate(spec,emb_dim,X,Xf,yf,Xv,yv,seed):
    actual=options.forced_seed if options.forced_seed is not None else seed
    m=original_candidate(spec,emb_dim,X,Xf,yf,Xv,yv,actual)
    # Re-select ELM hyperparameters using worst-session accuracy then mean.
    Ef,Ev=m['embed'](Xf),m['embed'](Xv)
    best=None
    for scale in (.25,.5,1.,2.):
        mul,add=em.emb_norm(Ef,scale)
        for ridge in (.1,1.,10.):
            beta=elm_fit(em.elm_input(Ef,mul,add),yf,ridge)
            pred=em.mcu_reference(em.elm_input(Ev,mul,add),beta).argmax(1)
            per=[float(np.mean(pred[groups==g]==yv[groups==g])) for g in range(len(validation))]
            key=(min(per),float(np.mean(per))) if options.robust_validation else (float(np.mean(pred==yv)),)
            if best is None or key>best[0]: best=(key,scale,ridge,per)
    key,scale,ridge,per=best
    m.update(s_elm=scale,lam=ridge,val_acc=key[0]+(key[1]*1e-4 if len(key)>1 else 0))
    scores.append(dict(seed=actual,sessions=dict(zip(validation,per)),worst=min(per),mean=float(np.mean(per))))
    return m


eb.build_train=build_train
em.train_net=train_net
em.elm_fit=elm_fit
em.build_candidate=build_candidate
em.main()
Path(value('--output-dir'),'validation_selection.json').write_text(json.dumps(scores,indent=2))

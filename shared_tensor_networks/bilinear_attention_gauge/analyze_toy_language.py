"""Canonical reduction of trained two-head toy-language attention.

The uniform measure is corpus-free on every role-valid query/key pair.
Empirical pair and sequence measures use fresh synthetic training examples;
validation uses an independent generator and exhaustive combinations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from run import fet_als
from toy_factorial import weighted_trace_gauge
from train_toy_language import BilinearAttentionToy, batch, exhaustive


QUERY_IDS=np.array([6,7])
KEY_IDS=np.arange(6)


def rms(z):
    return z/np.sqrt(np.mean(z*z,axis=-1,keepdims=True)+np.finfo(z.dtype).eps)


def factors(model,head):
    e=model.embed.weight.detach().numpy()
    sl=slice(head*4,(head+1)*4)
    def qk(name,ids):
        return rms(e[ids]@getattr(model,name).weight.detach().numpy()[sl].T)
    u=qk("q1",QUERY_IDS)/4
    v=qk("k1",KEY_IDS)
    u2=qk("q2",QUERY_IDS)
    v2=qk("k2",KEY_IDS)
    b=u2@v2.T/4
    value=e[KEY_IDS]@model.v.weight.detach().numpy()[sl].T
    c=value@model.o.weight.detach().numpy()[:,sl].T@model.read.weight.detach().numpy().T
    # This quotient is exact on the entire role-valid domain (2 queries x 6
    # keys), not a random subset of tokens. The unused physical bond directions
    # cannot have a positive-definite WTG boundary on this toy task.
    qk1=u@v.T
    left,s,right=np.linalg.svd(qk1,full_matrices=False)
    rank=int(np.sum(s>s[0]*1e-10))
    root=np.sqrt(s[:rank])
    uq=left[:,:rank]*root
    vq=right[:rank].T*root
    assert np.max(np.abs(qk1-uq@vq.T))<1e-12
    return dict(u=uq,v=vq,b=b,c=c,rank=rank,physical_width=4)


def pair_table(head,x):
    return (head["u"]@x@head["v"].T*head["b"])[:,:,None]*head["c"][None,:,:]


def sequence_features(head,seq):
    q=seq[:,-1]-6
    k=seq[:,:-1]
    assert np.min(q)>=0 and np.max(q)<2
    assert np.min(k)>=0 and np.max(k)<6
    u=head["u"][q]
    v=head["v"][k]
    b=head["b"][q[:,None],k]
    c=head["c"][k]
    features=np.einsum("nr,nsu,ns,nso->nruo",u,v,b,c,optimize=True)
    return features.reshape(len(seq),-1,2)


def env_pair(head,measure):
    weights=measure*head["b"]**2*np.sum(head["c"]**2,axis=1)[None,:]
    e=np.einsum("ar,at,bs,bu,ab->rstu",head["u"],head["u"],
                head["v"],head["v"],weights,optimize=True)
    d=head["rank"]
    return e.reshape(d*d,d*d)


def env_sequence(head,seq):
    f=sequence_features(head,seq)
    return np.einsum("npo,nqo->pq",f,f,optimize=True)/len(seq)


def pair_measure(seq):
    q=seq[:,-1]-6
    k=seq[:,:-1]
    counts=np.zeros((2,6),dtype=np.float64)
    np.add.at(counts,(np.repeat(q,3),k.ravel()),1.)
    return counts/counts.sum()


def approx_tables(heads,xs):
    return [pair_table(head,x) for head,x in zip(heads,xs)]


def logits_from_tables(tables,seq):
    q=seq[:,-1]-6
    k=seq[:,:-1]
    return sum(np.sum(t[q[:,None],k],axis=1) for t in tables)


def scores(tables,seq,labels,attrs=None):
    logits=logits_from_tables(tables,seq)
    logits_t=torch.from_numpy(logits)
    labels_t=torch.from_numpy(labels)
    ce=F.cross_entropy(logits_t,labels_t,reduction="none").numpy()
    correct=np.argmax(logits,axis=-1)==labels
    result=dict(accuracy=float(np.mean(correct)),ce=float(np.mean(ce)))
    if attrs is not None:
        for key,mask in (
            ("common",np.array([a["Q"]==0 for a in attrs])),
            ("rare_seen",np.array([a["Q"]==1 and not a["heldout"] for a in attrs])),
            ("rare_heldout",np.array([a["heldout"] for a in attrs]))):
            result[key]=dict(n=int(mask.sum()),accuracy=float(np.mean(correct[mask])),
                             ce=float(np.mean(ce[mask])))
    return result,logits


def one_seed(seed,n_fit=100000,n_val=50000):
    torch.set_num_threads(1)
    model=BilinearAttentionToy().double()
    checkpoint=Path(__file__).with_name(f"toy_language_seed{seed}.pt")
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
    model.eval()
    heads=[factors(model,h) for h in range(2)]
    original=approx_tables(heads,[np.eye(h["rank"]) for h in heads])
    seq_t,labels_t,attrs=exhaustive()
    seq=seq_t.numpy();labels=labels_t.numpy()
    direct=model(seq_t).detach().numpy()
    exact=logits_from_tables(original,seq)
    direct_gate=float(np.max(np.abs(direct-exact)))
    assert direct_gate<1e-10,direct_gate
    fit_seq,_=batch(n_fit,torch.Generator().manual_seed(100000+seed))
    check_seq,_=batch(n_fit,torch.Generator().manual_seed(300000+seed))
    val_seq,val_labels=batch(n_val,torch.Generator().manual_seed(200000+seed))
    fit_seq=fit_seq.numpy()
    check_seq=check_seq.numpy()
    val_seq,val_labels=val_seq.numpy(),val_labels.numpy()
    measures={"uniform_pair":np.ones((2,6))/12,
              "empirical_pair":pair_measure(fit_seq)}
    environments={
        "uniform_pair":[env_pair(h,measures["uniform_pair"]) for h in heads],
        "empirical_pair":[env_pair(h,measures["empirical_pair"]) for h in heads],
        "empirical_sequence":[env_sequence(h,fit_seq) for h in heads],
    }
    environment_comparison=[]
    for h in range(2):
        pair_env=environments["empirical_pair"][h]
        seq_env=environments["empirical_sequence"][h]
        pair_check=env_pair(heads[h],pair_measure(check_seq))
        seq_check=env_sequence(heads[h],check_seq)
        direction=np.eye(heads[h]["rank"]).ravel()
        pair_norm=direction@pair_env@direction
        seq_norm=direction@seq_env@direction
        pair_scaled=pair_env/pair_norm
        seq_scaled=seq_env/seq_norm
        pair_check_scaled=pair_check/(direction@pair_check@direction)
        seq_check_scaled=seq_check/(direction@seq_check@direction)
        environment_comparison.append(dict(head=h,
            sequence_vs_pair_relative_frobenius=float(np.linalg.norm(seq_scaled-pair_scaled)/
                                                      np.linalg.norm(pair_scaled)),
            pair_sampling_relative_frobenius=float(np.linalg.norm(pair_check_scaled-pair_scaled)/
                                                   np.linalg.norm(pair_scaled)),
            sequence_sampling_relative_frobenius=float(np.linalg.norm(seq_check_scaled-seq_scaled)/
                                                       np.linalg.norm(seq_scaled))))
    records=[]
    baseline,baseline_logits=scores(original,seq,labels,attrs)
    validation,_=scores(original,val_seq,val_labels)
    ablations=[]
    for removed in range(2):
        modified=[t.copy() for t in original]
        modified[removed]=np.zeros_like(modified[removed])
        ablation_ex,_=scores(modified,seq,labels,attrs)
        ablation_val,_=scores(modified,val_seq,val_labels)
        ablations.append(dict(removed_head=removed,exhaustive=ablation_ex,
                              validation=ablation_val))
    for measure_name,envs in environments.items():
        for method in ("wtg","fet"):
            xs=[]
            for env,h in zip(envs,heads):
                d=h["rank"]
                assert d==2,d
                wtg,s,gx,gy=weighted_trace_gauge(env,1)
                if method=="wtg":
                    x=wtg
                else:
                    x,_=fet_als(env,wtg,1,iterations=300)
                xs.append(x)
            tables=approx_tables(heads,xs)
            exhaustive_score,logits=scores(tables,seq,labels,attrs)
            val_score,val_logits=scores(tables,val_seq,val_labels)
            denom=np.mean(baseline_logits**2)
            records.append(dict(measure=measure_name,method=method,
                pair_measure=measures.get(measure_name,None).tolist() if measure_name in measures else None,
                exhaustive=exhaustive_score,validation=val_score,
                exhaustive_logit_mse=float(np.mean((logits-baseline_logits)**2)/denom),
                validation_logit_mse=float(np.mean((val_logits-
                    logits_from_tables(original,val_seq))**2)/
                    np.mean(logits_from_tables(original,val_seq)**2))))
    return dict(seed=seed,n_fit=n_fit,n_val=n_val,
                direct_table_max_abs=direct_gate,
                head_quotient_ranks=[h["rank"] for h in heads],
                baseline_exhaustive=baseline,baseline_validation=validation,
                head_ablations=ablations,
                environment_comparison=environment_comparison,
                records=records)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--n-fit",type=int,default=100000)
    p.add_argument("--n-val",type=int,default=50000)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    result=one_seed(args.seed,args.n_fit,args.n_val)
    path=args.output or Path(__file__).with_name(f"toy_language_analysis_seed{args.seed}.json")
    path.write_text(json.dumps(result,indent=2)+"\n")
    for row in result["records"]:
        print(args.seed,row["measure"],row["method"],
              row["exhaustive"]["accuracy"],row["exhaustive"]["rare_heldout"],
              row["validation"]["ce"],flush=True)

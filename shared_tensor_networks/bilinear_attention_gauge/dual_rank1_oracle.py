"""Supervised rank-one capacity and fidelity tradeoffs on the dual-output toy.

The exhaustive mode uses held-out labels as an oracle capacity ceiling.
The train mode excludes them and selects starts on independent training-law
validation; neither mode is an unsupervised canonicalization method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from analyze_dual_output_language import factors,logits,score
from analyze_toy_language import pair_table
from train_dual_output_language import DualOutputAttention,batch,exhaustive


def run(seed,starts=8,steps=3000,mse_weight=0.,fit_domain="exhaustive",
        batch_size=512,fidelity_domain="train_sequence"):
    torch.set_num_threads(1)
    model=DualOutputAttention().double()
    model.load_state_dict(torch.load(Path(__file__).with_name(f"dual_output_seed{seed}.pt"),
                                     map_location="cpu",weights_only=True))
    heads=[factors(model,h) for h in range(2)]
    seq_t,targets_t,attrs=exhaustive()
    seq,targets=seq_t.numpy(),targets_t.double()
    q=seq[:,-1]-6
    k=seq[:,:-1]
    def tensors_for(sequences):
        queries=sequences[:,-1]-6
        keys=sequences[:,:-1]
        return [tuple(torch.from_numpy(x) for x in
            (h["u"][queries],h["v"][keys],h["b"][queries[:,None],keys],h["c"][keys]))
                for h in heads]
    exhaustive_tensors=tensors_for(seq)
    bias=model.read.bias.detach()
    original=[pair_table(h,np.eye(2)) for h in heads]
    original_logits=torch.from_numpy(logits(original,seq,bias.numpy()))
    val_seq_t,val_targets_t=batch(50000,torch.Generator().manual_seed(500000+seed))
    val_seq=val_seq_t.numpy()
    val_targets=val_targets_t.double()
    val_tensors=tensors_for(val_seq)
    val_original_logits=torch.from_numpy(logits(original,val_seq,bias.numpy()))
    def forward(tensors,n,params):
        out=bias[None,:].expand(n,-1).clone()
        for h,(u,v,b,c) in enumerate(tensors):
            x=torch.outer(params[2*h],params[2*h+1])
            s=torch.einsum("nr,rs,nks->nk",u,x,v)
            out=out+torch.sum((s*b)[:,:,None]*c,dim=1)
        return out
    best=None
    for start in range(starts):
        torch.manual_seed(seed*1000+start+71)
        generator=torch.Generator().manual_seed(seed*10000+start+81)
        params=[torch.nn.Parameter(torch.randn(2,dtype=torch.float64))
                for _ in range(4)]
        optimizer=torch.optim.Adam(params,lr=.025)
        for _ in range(steps):
            if fit_domain=="exhaustive":
                fit_tensors=exhaustive_tensors
                fit_targets=targets
                fit_original=original_logits
            elif fit_domain=="train":
                sample_seq,sample_targets=batch(batch_size,generator)
                sample_seq=sample_seq.numpy()
                fit_tensors=tensors_for(sample_seq)
                fit_targets=sample_targets.double()
                fit_original=torch.from_numpy(logits(original,sample_seq,bias.numpy()))
            else:
                raise ValueError(fit_domain)
            out=forward(fit_tensors,len(fit_targets),params)
            bce=F.binary_cross_entropy_with_logits(out,fit_targets)
            if fidelity_domain=="train_sequence":
                relative_mse=torch.mean((out-fit_original)**2)/torch.mean(fit_original**2)
            elif fidelity_domain=="uniform_exhaustive":
                uniform_out=forward(exhaustive_tensors,len(seq),params)
                relative_mse=torch.mean((uniform_out-original_logits)**2)/torch.mean(original_logits**2)
            else:
                raise ValueError(fidelity_domain)
            loss=bce+mse_weight*relative_mse
            optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():
            xs=[torch.outer(params[2*h],params[2*h+1]).numpy().copy() for h in range(2)]
            table=[pair_table(h,x) for h,x in zip(heads,xs)]
            result,approx_logits=score(table,seq,targets.numpy(),bias.numpy(),attrs)
            mse=float(np.mean((approx_logits-original_logits.numpy())**2)/
                      np.mean(original_logits.numpy()**2))
            candidate=dict(start=start,score=result,relative_logit_mse=mse,
                           matrices=[x.tolist() for x in xs])
            if fit_domain=="train":
                val_out=forward(val_tensors,len(val_seq),params)
                val_bce=float(F.binary_cross_entropy_with_logits(val_out,val_targets))
                val_mse=float(torch.mean((val_out-val_original_logits)**2)/
                              torch.mean(val_original_logits**2))
                fidelity_for_selection=(val_mse if fidelity_domain=="train_sequence"
                                        else mse)
                candidate["validation_bce"]=val_bce
                candidate["validation_relative_logit_mse"]=val_mse
                candidate["selection_loss"]=val_bce+mse_weight*fidelity_for_selection
            else:
                candidate["selection_loss"]=result["bce"]+mse_weight*mse
            if best is None or candidate["selection_loss"]<best["selection_loss"]:
                best=candidate
    return dict(seed=seed,starts=starts,steps=steps,mse_weight=mse_weight,
                fit_domain=fit_domain,batch_size=batch_size,
                fidelity_domain=fidelity_domain,
                scope=("supervised all-96 oracle, including held-out combinations" if
                       fit_domain=="exhaustive" else
                       "supervised training distribution, held-out rare combination excluded"),
                best=best)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--starts",type=int,default=8)
    p.add_argument("--steps",type=int,default=3000)
    p.add_argument("--mse-weight",type=float,default=0.)
    p.add_argument("--fit-domain",choices=("exhaustive","train"),default="exhaustive")
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--fidelity-domain",choices=("train_sequence","uniform_exhaustive"),
                   default="train_sequence")
    args=p.parse_args()
    result=run(args.seed,args.starts,args.steps,args.mse_weight,args.fit_domain,
               args.batch_size,args.fidelity_domain)
    path=Path(__file__).with_name(
        f"dual_rank1_{args.fit_domain}_seed{args.seed}_mse{args.mse_weight:g}"
        f"_{args.fidelity_domain}.json")
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(args.seed,result["best"]["score"],result["best"]["relative_logit_mse"])

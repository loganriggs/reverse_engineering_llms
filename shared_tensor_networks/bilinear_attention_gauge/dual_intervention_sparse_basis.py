"""Label-free sparse gauges inside intervention-aware key subspaces."""

from __future__ import annotations

import json

import numpy as np
import scipy.optimize as so
import torch

from dual_intervention_dictionary import tables_for_model
from dual_joint_function_canonical import HERE,TRUTH,unflatten
from train_dual_output_language import DualOutputAttention


def alignment(dictionary):
    similarity=np.abs(dictionary@TRUTH.T)
    similarity/=np.linalg.norm(dictionary,axis=1,keepdims=True)
    row,col=so.linear_sum_assignment(-similarity)
    return float(np.mean(similarity[row,col])),{("A" if c==0 else "B"):int(r)
                                            for r,c in zip(row,col)}


def random_rotation(rank,rng):
    q,r=np.linalg.qr(rng.standard_normal((rank,rank)))
    return q@np.diag(np.sign(np.diag(r)))


def sparse_rotation(dictionary,seed,starts=16,sweeps=12,grid=256):
    rng=np.random.default_rng(10000+seed)
    rank=len(dictionary)
    angles=np.linspace(0,np.pi,grid,endpoint=False)
    cs=np.cos(angles);ss=np.sin(angles)
    best=None
    best_loss=np.inf
    for start in range(starts):
        rot=np.eye(rank) if start==0 else random_rotation(rank,rng)
        current=rot@dictionary
        for _ in range(sweeps):
            changed=False
            for a in range(rank):
                for b in range(a+1,rank):
                    x=current[a];y=current[b]
                    first=cs[:,None]*x-ss[:,None]*y
                    second=ss[:,None]*x+cs[:,None]*y
                    losses=np.abs(first).sum(1)+np.abs(second).sum(1)
                    chosen=int(np.argmin(losses))
                    old=np.abs(x).sum()+np.abs(y).sum()
                    if losses[chosen]<old-1e-10:
                        c=cs[chosen];s=ss[chosen]
                        current[[a,b]]=np.stack((c*x-s*y,s*x+c*y))
                        rot[[a,b]]=np.stack((c*rot[a]-s*rot[b],
                                              s*rot[a]+c*rot[b]))
                        changed=True
            if not changed: break
        loss=float(np.abs(current).sum())
        if loss<best_loss:
            best_loss=loss;best=(rot,current,start)
    rot,current,start=best
    assert np.max(np.abs(rot@rot.T-np.eye(rank)))<1e-10
    assert np.max(np.abs(current-rot@dictionary))<1e-10
    return rot,current,dict(l1=best_loss,selected_start=start)


def original_component_effects(original_heads,dictionary):
    # Refit head-specific coefficients after gauge selection. Sum both heads
    # to test whether the matched component has selective A/B token effects.
    projected=original_heads@dictionary.T
    merged=projected[:4]+projected[4:]
    result=[]
    for index in range(len(dictionary)):
        table=unflatten(np.outer(merged[:,index],dictionary[index]))
        energy={"A":float(np.sum((table[:,1,:]-table[:,0,:])**2)),
                "B":float(np.sum((table[:,3,:]-table[:,2,:])**2)),
                "F":float(np.sum((table[:,5,:]-table[:,4,:])**2))}
        total=sum(energy.values())
        dominant=max(energy,key=energy.get)
        result.append(dict(component=index,energy=energy,dominant=dominant,
                           dominant_fraction=energy[dominant]/max(total,1e-30),
                           total_bit_flip_energy=total))
    return result


def weighted_purity(effects):
    total=sum(e["total_bit_flip_energy"] for e in effects)
    return float(sum(e["dominant_fraction"]*e["total_bit_flip_energy"]
                     for e in effects)/max(total,1e-30))


def run():
    torch.set_num_threads(1)
    configurations={"original_rank2":(["original"],2),
                    "three_rank4":(["original","qk1_constant","qk2_constant"],4),
                    "four_rank4":(["original","qk1_constant","qk2_constant","both_constant"],4),
                    "four_rank5":(["original","qk1_constant","qk2_constant","both_constant"],5)}
    rows=[]
    for seed in range(10):
        model=DualOutputAttention().double()
        model.load_state_dict(torch.load(HERE/f"dual_output_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        model.eval()
        tables=tables_for_model(model)
        for name,(seen,rank) in configurations.items():
            stack=np.vstack([tables[k]/np.linalg.norm(tables[k]) for k in seen])
            _,_,v=np.linalg.svd(stack,full_matrices=False)
            dictionary=v[:rank]
            rot,sparse,optimization=sparse_rotation(dictionary,seed+100*rank)
            assert np.max(np.abs(dictionary.T@dictionary-sparse.T@sparse))<1e-10
            contrast_subspace_cosine=float(np.mean(
                np.linalg.norm(TRUTH@(dictionary.T@dictionary),axis=1)))
            rng=np.random.default_rng(seed+1000+rank)
            random_align=[];random_purity=[]
            for _ in range(128):
                random_d=random_rotation(rank,rng)@dictionary
                random_align.append(alignment(random_d)[0])
                random_purity.append(weighted_purity(original_component_effects(
                    tables["original"],random_d)))
            variants={"svd":dictionary,"sparse":sparse}
            result={}
            for variant,d in variants.items():
                align,match=alignment(d)
                result[variant]=dict(l1=float(np.abs(d).sum()),alignment=align,
                    matched_components=match,dictionary=d.tolist(),
                    original_component_effects=original_component_effects(
                        tables["original"],d))
                result[variant]["original_component_weighted_purity"]=\
                    weighted_purity(result[variant]["original_component_effects"])
            rows.append(dict(seed=seed,configuration=name,rank=rank,
                optimization=optimization,
                contrast_subspace_cosine=contrast_subspace_cosine,
                random_alignment_mean=float(np.mean(random_align)),
                random_alignment_p95=float(np.quantile(random_align,.95)),
                random_purity_mean=float(np.mean(random_purity)),
                results=result))
    output=dict(n_seeds=10,
        fitting="SVD of exact token tables, followed by key L1 orthogonal rotation; no task labels",
        evaluation="post-fit A/B contrast alignment and original-function bit-flip selectivity",
        random_baseline="128 Haar orthogonal rotations per seed and rank",
        rows=rows)
    (HERE/"dual_intervention_sparse_basis.json").write_text(json.dumps(output,indent=2)+"\n")
    for name in configurations:
        selected=[r for r in rows if r["configuration"]==name]
        print(name,*[round(float(np.mean([r["results"][kind]["alignment"]
                       for r in selected])),4) for kind in ("svd","sparse")],
              round(float(np.mean([r["random_alignment_mean"] for r in selected])),4),
              flush=True)
    return output


if __name__=="__main__":
    run()

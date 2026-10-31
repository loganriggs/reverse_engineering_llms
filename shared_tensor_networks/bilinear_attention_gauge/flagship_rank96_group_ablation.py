"""Full-model effects of energy-selected stable rank-96 QK1 groups."""

from __future__ import annotations

import json
import argparse

import numpy as np
import torch

from flagship_component_ablation import calibrate,kl_per_document
from flagship_full_model_eval import BondEdit,GPT,GPTConfig,get_cuts,logits
from flagship_matrix_free import CORPUS,HD,SNAPSHOT,rows
from flagship_ood_corpora import DATASETS
from flagship_rank96_group_stability import CUTS,selection_order
from flagship_transfer import ART


def candidates(state,cut_name,group_size=4,n_selection=16384,
               group_indices=(0,)):
    fineweb=np.load(CORPUS,mmap_mode="r")
    _,selection,_=rows(state,0,71,1,n_selection,128,
        distribution="corpus",corpus=fineweb,val_docs=(15000,17500))
    base=get_cuts()[cut_name]
    left,s,right=np.linalg.svd(base,full_matrices=False)
    rank=int(np.linalg.matrix_rank(base))
    a=left[:,:rank]*np.sqrt(s[:rank]);b=right[:rank].T*np.sqrt(s[:rank])
    keys=[("svd",None,None)]
    keys += [("random",None,seed) for seed in (0,1,2)]
    keys += [("intervention_activation","identity",seed) for seed in (0,1,2)]
    keys += [("intervention_activation","random_start",seed) for seed in (0,1,2)]
    rotation_files={family:np.load(ART/f"head0_exact_gauge_rotations_{cut_name}"
                              f"{suffix}.npz")
                    for family,suffix in (("identity",""),("random_start","_random_start"))}
    edits={};details={}
    for kind,family,seed in keys:
        if kind=="svd":
            rotation=np.eye(rank);name="svd"
        elif kind=="random":
            rotation=rotation_files["identity"][f"random_{seed}"]
            name=f"random_{seed}"
        else:
            rotation=rotation_files[family][f"{kind}_{seed}"]
            name=f"{family}_{seed}"
        order=selection_order(selection,a,b,rotation)
        for group_index in group_indices:
            components=order[group_size*group_index:group_size*(group_index+1)]
            basis=rotation[:,components]
            effect=(a@basis)@(b@basis).T
            assert np.linalg.norm(effect)/np.linalg.norm(base)>0
            group_name=name if group_indices==(0,) else f"{name}_g{group_index}"
            edits[group_name]=base-effect
            details[group_name]=dict(kind=kind,start=family,seed=seed,
                group_index=group_index,group_size=group_size,
                components=components.tolist())
    return base,edits,details


def run(n_docs=4,seeds=(23,29),batch_size=4,group_size=4,
        target_head_error=.001,calibration_docs=64,
        group_indices=(0,)):
    torch.set_num_threads(4)
    config=json.loads((SNAPSHOT/"config.json").read_text())
    config.pop("step",None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    model.load_state_dict(state)
    editor=BondEdit(model)
    corpora={"fineweb_test":(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[];all_details={};calibrations={};identity_gate=0.
    with torch.no_grad():
        for cut_name in CUTS:
            base,edits,details=candidates(state,cut_name,group_size,
                                           group_indices=group_indices)
            edits,calibration=calibrate(state,base,edits,details,
                                         calibration_docs,target_head_error)
            all_details[cut_name]=details;calibrations[cut_name]=calibration
            base_t=torch.from_numpy(base).to(torch.float32)
            edited={name:torch.from_numpy(cut).to(torch.float32)
                    for name,cut in edits.items()}
            for corpus_name,(path,begin,end) in corpora.items():
                corpus=np.load(path,mmap_mode="r")
                for seed in seeds:
                    rng=np.random.default_rng(seed+128000)
                    docs=rng.choice(np.arange(begin,end),size=n_docs,replace=False)
                    for offset in range(0,n_docs,batch_size):
                        spans=[]
                        for doc in docs[offset:offset+batch_size]:
                            start=int(rng.integers(0,corpus.shape[1]-129+1))
                            spans.append(np.asarray(corpus[doc,start:start+129],
                                                    dtype=np.int64))
                        batch=np.stack(spans)
                        idx=torch.from_numpy(batch[:,:-1].copy())
                        if not records:
                            editor.cut=None
                            native=logits(model,idx)
                            editor.cut=torch.eye(HD)
                            identity_gate=float((native-logits(model,idx)).abs().max())
                            assert identity_gate<1e-5,identity_gate
                        editor.cut=base_t
                        reference=logits(model,idx)
                        for name,cut in edited.items():
                            editor.cut=cut
                            effect=kl_per_document(reference,logits(model,idx))
                            for position,doc in enumerate(docs[offset:offset+batch_size]):
                                records.append(dict(cut=cut_name,corpus=corpus_name,
                                    seed=seed,document_id=int(doc),
                                    document_position=offset+position,
                                    group=name,base_to_removed_kl=float(effect[position])))
                    print(cut_name,corpus_name,seed,flush=True)
    output=dict(layer=0,head=0,group_size=group_size,
        group_indices=list(group_indices),n_docs=n_docs,
        seeds=list(seeds),target_calibration_head_error=target_head_error,
        calibration_documents=calibration_docs,
        selection_fineweb_docs=[15000,17500],
        test_fineweb_docs=[17500,20000],
        identity_gate_max_abs_logit=identity_gate,
        scope="calibrated group partial deletion, full-model per-document KL from each rank-96 base cut",
        details=all_details,calibrations=calibrations,records=records)
    suffix="" if group_indices==(0,) else "_groups"+"_".join(map(str,group_indices))
    (ART/f"head0_rank96_group_ablation_n{n_docs}{suffix}.json").write_text(
        json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--n-docs",type=int,default=8)
    parser.add_argument("--groups",type=int,nargs="+",default=[0])
    args=parser.parse_args()
    run(n_docs=args.n_docs,group_indices=tuple(args.groups))

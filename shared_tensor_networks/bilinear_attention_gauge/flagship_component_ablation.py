"""Full-model effects of removing high-energy components in exact gauges.

Compares KL from a fixed rank-64 cut to one-component-removed cuts. Reports
per-document distribution of effects, not semantic labels or task discovery.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_full_model_eval import BondEdit,GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_sequence_transfer import sequence_head
from flagship_transfer import ART


GAUGES=("svd","random_0","factor_weight_0","activation_0",
        "intervention_activation_0","rare_balanced_activation_0")


def selected_cuts(top_components=2):
    x=np.load(ART/"head0_l128_sequence_fit_rank64_cw0.npz")["cut"]
    left,s,right=np.linalg.svd(x,full_matrices=False)
    a=left[:,:64]*np.sqrt(s[:64]);b=right[:64].T*np.sqrt(s[:64])
    rotations=np.load(ART/"head0_exact_gauge_rotations.npz")
    label_data=json.loads((ART/"head0_component_label_transfer.json").read_text())
    cuts={};details={}
    for gauge in GAUGES:
        r=rotations[gauge]
        factors_a=a@r;factors_b=b@r
        assert np.linalg.norm(factors_a@factors_b.T-x)/np.linalg.norm(x)<1e-10
        records=sorted((z for z in label_data["records"]
                        if z["corpus"]=="selection" and z["gauge"]==gauge),
                       key=lambda z:z["selection_rank"])
        for rec in records[:top_components]:
            component=rec["component"]
            name=f"{gauge}_rank{rec['selection_rank']}_c{component}"
            cuts[name]=x-np.outer(factors_a[:,component],factors_b[:,component])
            details[name]=dict(gauge=gauge,component=component,
                selection_rank=rec["selection_rank"],
                proposed_token_form_label=rec["proposed_label"])
    return x,cuts,details


def kl_per_document(reference,candidate):
    p=F.log_softmax(reference.to(torch.float64),dim=-1)
    q=F.log_softmax(candidate.to(torch.float64),dim=-1)
    return (p.exp()*(p-q)).sum(-1).mean(-1).numpy()


def calibrate(state,base,cuts,details,n_docs,target_error):
    corpus=np.load(CORPUS,mmap_mode="r")
    rng=np.random.default_rng(811)
    docs=rng.choice(np.arange(15000,17500),size=n_docs,replace=False)
    total_energy=0.
    sse={name:0. for name in cuts}
    for doc in docs:
        start=int(rng.integers(0,corpus.shape[1]-128+1))
        tokens=np.asarray(corpus[doc,start:start+128],dtype=np.int64)
        outputs=sequence_head(state,tokens,0,{"base":base,**cuts})
        original=outputs["base"]
        total_energy+=float(np.sum(original**2))
        for name in cuts:
            sse[name]+=float(np.sum((outputs[name]-original)**2))
    scaled={}
    for name,cut in cuts.items():
        error=sse[name]/total_energy
        factor=float(np.sqrt(target_error/error))
        if factor>1.:
            raise ValueError(f"target too large for deletion-only edit {name}: {factor}")
        scaled[name]=base+factor*(cut-base)
        details[name]["calibration_head_error_full_removal"]=error
        details[name]["component_deletion_fraction"]=factor
    return scaled,dict(n_docs=n_docs,seed=811,
        fineweb_documents=[15000,17500],target_relative_head_error=target_error)


def run(n_docs=8,seeds=(23,29),batch_size=4,threads=2,
        top_components=2,target_head_error=None,calibration_docs=64):
    torch.set_num_threads(threads)
    config=json.loads((SNAPSHOT/"config.json").read_text())
    config.pop("step",None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    model.load_state_dict(state)
    editor=BondEdit(model)
    base_np,cut_np,details=selected_cuts(top_components)
    calibration=None
    if target_head_error is not None:
        cut_np,calibration=calibrate(state,base_np,cut_np,details,
                                     calibration_docs,target_head_error)
    base=torch.from_numpy(base_np).to(torch.float32)
    cuts={name:torch.from_numpy(cut).to(torch.float32)
          for name,cut in cut_np.items()}
    corpora={"fineweb_test":(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[];gate=None
    with torch.no_grad():
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
                    target=torch.from_numpy(batch[:,1:].copy())
                    editor.cut=None
                    original=logits(model,idx)
                    if gate is None:
                        native=model(idx,target)
                        expected=F.cross_entropy(original.reshape(-1,original.shape[-1]),
                                                 target.ravel())
                        gate=float(abs(native-expected))
                        assert gate<1e-5,gate
                    editor.cut=base
                    reference=logits(model,idx)
                    base_kl=kl_per_document(original,reference)
                    for name,cut in cuts.items():
                        editor.cut=cut
                        candidate=logits(model,idx)
                        effect=kl_per_document(reference,candidate)
                        original_kl=kl_per_document(original,candidate)
                        for index,doc in enumerate(docs[offset:offset+batch_size]):
                            tokens=batch[index,:-1]
                            outputs=sequence_head(state,tokens,0,
                                {"base":base_np,"removed":cut_np[name]})
                            denominator=float(np.sum(outputs["base"]**2))
                            head_error=float(np.sum((outputs["removed"]-
                                                     outputs["base"])**2)/denominator)
                            records.append(dict(corpus=corpus_name,seed=seed,
                                document_id=int(doc),batch_offset=offset+index,
                                cut=name,base_to_removed_kl=float(effect[index]),
                                original_to_base_kl=float(base_kl[index]),
                                original_to_removed_kl=float(original_kl[index]),
                                relative_head_output_error=head_error))
                print(corpus_name,seed,flush=True)
    result=dict(head=0,layer=0,rank=64,length=128,
        n_docs_per_seed=n_docs,seeds=list(seeds),batch_size=batch_size,
        top_components_per_gauge=top_components,
        native_ce_gate=gate,
        component_selection="top pair-response energy on FineWeb label-selection sample; no semantic labels used",
        calibration=calibration,
        selection_documents=[15000,17500],fineweb_test_documents=[17500,20000],
        scope="full-model per-document component ablation KL; no semantic selectivity claim",
        details=details,records=records)
    suffix="_matched" if target_head_error is not None else ""
    (ART/f"head0_component_ablation_n{n_docs}{suffix}.json").write_text(
        json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--n-docs",type=int,default=8)
    parser.add_argument("--seeds",type=int,nargs="+",default=[23,29])
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--top-components",type=int,default=2)
    parser.add_argument("--target-head-error",type=float,default=None)
    parser.add_argument("--calibration-docs",type=int,default=64)
    args=parser.parse_args()
    run(args.n_docs,tuple(args.seeds),args.batch_size,args.threads,
        args.top_components,args.target_head_error,args.calibration_docs)

"""Exploratory outlier follow-up: ' as' -> ' much' and matched windows.

The motif was selected after inspecting one FineWeb test outlier. It is not a
pre-registered semantic validation. It asks whether the high head-output edit
on that pair recurs on other documents and external corpora.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from flagship_component_ablation import selected_cuts
from flagship_matrix_free import CORPUS,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_sequence_transfer import sequence_head
from flagship_transfer import ART


AS=355
MUCH=881
GAUGES=("svd_rank1_c1","random_0_rank0_c31",
        "activation_0_rank1_c15","key_uniform_0_rank0_c56",
        "key_corpus_0_rank0_c1")


def windows(corpus,begin,end,seed,max_motif=32):
    rng=np.random.default_rng(seed)
    sub=np.asarray(corpus[begin:end])
    doc,pos=np.where((sub[:,:-1]==AS)&(sub[:,1:]==MUCH))
    long_document=len(sub[0])>129
    candidate=[(int(begin+d),int(p)) for d,p in zip(doc,pos)
               if ((not long_document and p+1<128) or
                   (long_document and p>=7 and p+121<len(sub[0])))]
    rng.shuffle(candidate)
    chosen=[];seen=set()
    for item in candidate:
        if item[0] not in seen:
            chosen.append(item);seen.add(item[0])
        if len(chosen)>=max_motif:break
    result=[]
    for d,p in chosen:
        start=p-7 if long_document else 0
        motif=np.asarray(corpus[d,start:start+128],dtype=np.int64)
        query_position=p-start+1
        assert query_position<128 and motif[query_position-1]==AS and motif[query_position]==MUCH
        # Draw a same-document control whose 128-token span lacks the motif.
        choices=[]
        for control_start in range(0,corpus.shape[1]-128+1):
            span=corpus[d,control_start:control_start+128]
            if not np.any((span[:-1]==AS)&(span[1:]==MUCH)):
                choices.append(control_start)
        if choices:
            control_doc=d;control_start=int(rng.choice(choices))
        else:
            available=np.flatnonzero(~np.any((sub[:,:-1]==AS)&(sub[:,1:]==MUCH),axis=1))
            control_doc=int(begin+rng.choice(available));control_start=0
        control=np.asarray(corpus[control_doc,control_start:control_start+128],dtype=np.int64)
        result.extend((dict(doc=d,position=p,start=start,query_position=query_position,
                            kind="motif",tokens=motif),
                       dict(doc=control_doc,position=p,start=control_start,
                            query_position=query_position,kind="control",tokens=control)))
    return result,dict(total_motif_occurrences=len(candidate),
                       unique_motif_documents=len(set(d for d,p in candidate)),
                       selected_motif_documents=len(result)//2)


def run():
    torch.set_num_threads(1)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    base,full_cuts,_=selected_cuts(2,"key")
    prior=json.loads((ART/"head0_component_ablation_n8_matched_key_panel.json").read_text())
    cuts={name:base+prior["details"][name]["component_deletion_fraction"]*
          (full_cuts[name]-base) for name in GAUGES}
    corpora={"fineweb_test":(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[];counts={}
    for name,(path,begin,end) in corpora.items():
        corpus=np.load(path,mmap_mode="r")
        cases,counts[name]=windows(corpus,begin,end,91000+begin,max_motif=32)
        for case in cases:
            outputs=sequence_head(state,case["tokens"],0,
                                  {"base":base,**cuts})
            denominator=float(np.sum(outputs["base"]**2))
            for gauge in cuts:
                delta=np.sum((outputs[gauge]-outputs["base"])**2,axis=1)
                records.append(dict(corpus=name,document_id=case["doc"],
                    motif_document_position=case["position"],
                    window_start=case["start"],kind=case["kind"],gauge=gauge,
                    relative_head_error=float(delta.sum()/denominator),
                    query_position_error=float(delta[case["query_position"]]),
                    max_position_error=float(delta.max())))
        print(name,counts[name],flush=True)
    output=dict(motif_token_ids=[AS,MUCH],
        motif="' as' followed by ' much', selected after inspecting one FineWeb test outlier",
        pairing="FineWeb motif pair at positions 7-8 with same-document motif-free control; 129-token external documents use natural pair position and motif-free control from same document if possible, otherwise another document",
        cuts="precalibrated partial deletions from matched full-model ablation; no refit",
        counts=counts,records=records)
    (ART/"head0_key_sparse_motif_followup.json").write_text(json.dumps(output,indent=2)+"\n")
    for corpus_name in corpora:
        for gauge in GAUGES:
            subset=[r for r in records if r["corpus"]==corpus_name and
                    r["gauge"]==gauge]
            if not subset:continue
            grouped={kind:np.array([r["relative_head_error"] for r in subset
                                    if r["kind"]==kind])
                     for kind in ("motif","control")}
            print(corpus_name,gauge,
                "motif",float(np.median(grouped["motif"])),
                "control",float(np.median(grouped["control"])),flush=True)
    return output


if __name__=="__main__":run()

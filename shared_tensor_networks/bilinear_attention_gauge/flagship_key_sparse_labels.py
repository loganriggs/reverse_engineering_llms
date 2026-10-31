"""Post-fit token-form labels for new key-side gauges and matched controls."""

from __future__ import annotations

import json

import numpy as np
import torch

from flagship_component_labels import (CATEGORIES,Decoder,check_decoder,
    distribution,label_name,sample)
from flagship_matrix_free import CORPUS,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART


def run(n_pairs=32768,top_components=10):
    torch.set_num_threads(1)
    decoder=Decoder();check_decoder(decoder)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    x=np.load(ART/"head0_l128_sequence_fit_rank64_cw0.npz")["cut"]
    left,s,right=np.linalg.svd(x,full_matrices=False)
    a=left[:,:64]*np.sqrt(s[:64]);b=right[:64].T*np.sqrt(s[:64])
    fineweb=np.load(CORPUS,mmap_mode="r")
    parts={"selection":sample(state,fineweb,15000,17500,n_pairs,71,a,b,decoder),
           "fineweb_test":sample(state,fineweb,17500,20000,n_pairs,73,a,b,decoder)}
    for name,path in DATASETS.items():
        corpus=np.load(path,mmap_mode="r")
        parts[name]=sample(state,corpus,0,1200,n_pairs,73,a,b,decoder)
    del state
    rotations=np.load(ART/"head0_key_sparse_rotations.npz")
    records=[]
    for gauge in rotations.files:
        rotation=rotations[gauge]
        selection,base,size,_=distribution(parts["selection"],rotation)
        eligible=base>=.02
        order=np.argsort(size)[::-1][:top_components]
        labels={}
        for component in order:
            enrichment=np.where(eligible,
                selection[component]/np.maximum(base,1e-30),-1)
            labels[int(component)]=int(np.argmax(enrichment))
        for corpus_name,part in parts.items():
            comp,base_test,_,_=distribution(part,rotation)
            for component,category in labels.items():
                fraction=float(comp[component,category])
                base_fraction=float(base_test[category])
                records.append(dict(gauge=gauge,corpus=corpus_name,
                    component=component,selection_rank=int(np.where(order==component)[0][0]),
                    proposed_label=label_name(category),
                    component_fraction=fraction,base_fraction=base_fraction,
                    log2_enrichment=float(np.log2(max(fraction,1e-30)/
                                                   max(base_fraction,1e-30)))))
    result=dict(head=0,rank=64,n_pairs_per_corpus=n_pairs,
        top_components=top_components,label_categories=list(CATEGORIES),
        label_rule="max selection enrichment among categories with >=2% original-response energy",
        selection_fineweb_documents=[15000,17500],
        test_fineweb_documents=[17500,20000],
        scope="post-fit coarse token-form labels; labels do not choose gauges",
        records=records)
    (ART/"head0_key_sparse_label_transfer.json").write_text(json.dumps(result,indent=2)+"\n")
    for corpus_name in ("fineweb_test",*DATASETS):
        print(corpus_name,flush=True)
        for kind in ("svd","random_0","activation_0","key_uniform_","key_corpus_"):
            selected=[r["log2_enrichment"] for r in records
                if r["corpus"]==corpus_name and r["gauge"].startswith(kind)]
            print(kind,round(float(np.mean(selected)),4),flush=True)
    return result


if __name__=="__main__":run()

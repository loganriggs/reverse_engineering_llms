"""Cross-corpus transfer of head-0 width-128 rank-16 cuts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from flagship_matrix_free import SNAPSHOT,rows
from flagship_transfer import ART,load_cuts
from matrix_free_environment import SampledBondEnvironment


OOD=Path("/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_ood_corpora")
DATASETS={"wikitext103":OOD/"wikitext103_1200x129.npy",
          "tinystories":OOD/"tinystories_1200x129.npy"}


def run(n_test=32768,seeds=(23,29)):
    torch.set_num_threads(1)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    cuts=load_cuts()
    records=[]
    for corpus_name,path in DATASETS.items():
        corpus=np.load(path,mmap_mode="r")
        for length in (4,128):
            for seed in seeds:
                _,validation,_=rows(state,0,seed,1,n_test,length,
                    distribution="corpus",corpus=corpus,
                    train_docs=(0,1),val_docs=(0,len(corpus)))
                u,v,w,b,c_norm=validation
                natural=SampledBondEnvironment(u,v,w/w.sum())
                constant=SampledBondEnvironment(u,v,c_norm/c_norm.sum())
                for name,x in cuts.items():
                    records.append(dict(corpus=corpus_name,length=length,
                        seed=seed,cut=name,pair_error=natural.relative_error(x),
                        qk2_constant_error=constant.relative_error(x)))
                print(corpus_name,length,seed,flush=True)
    result=dict(head=0,ranks=[16,32,64,96],n_test_per_cell=n_test,seeds=list(seeds),
                data_paths={k:str(v) for k,v in DATASETS.items()},
                scope="first-layer head pair contribution; not full model CE",
                records=records)
    output=ART/"head0_ood_corpora.json"
    output.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":run()

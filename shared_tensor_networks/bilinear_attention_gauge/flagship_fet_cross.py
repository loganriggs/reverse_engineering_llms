"""Matched rank-16, 65k-pair cross-measure evaluation of full-width fits."""

import json
from pathlib import Path

import numpy as np
import torch

from flagship_matrix_free import CORPUS,SNAPSHOT,rows
from matrix_free_environment import SampledBondEnvironment


def run():
    torch.set_num_threads(1)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    corpus=np.load(CORPUS,mmap_mode="r")
    environments={}
    for measure in ("uniform","corpus"):
        _,validation,_=rows(state,0,23,1,65536,4,
            distribution=measure,corpus=corpus if measure=="corpus" else None)
        u,v,w,_,_=validation
        environments[measure]=SampledBondEnvironment(u,v,w/w.sum())
    results=[]
    base=Path(__file__).with_name("full_width_results")
    for fit_measure in ("uniform","corpus"):
        suffix="_corpus" if fit_measure=="corpus" else ""
        stem=f"head0_seed17_n65536{suffix}"
        for method,path,keys in (
            ("wtg",base/(stem+".npz"),("wtg_rank16",)),
            ("local",base/(stem+".npz"),("local_rank16",)),
            ("direct_fit",base/(stem+"_fet.npz"),("wtg_rank16","local_rank16"))):
            arrays=np.load(path)
            for key in keys:
                x=arrays[key]
                for test_measure,env in environments.items():
                    results.append(dict(fit_measure=fit_measure,method=method,
                        start=key if method=="direct_fit" else None,
                        test_measure=test_measure,relative_squared_error=env.relative_error(x)))
    data=dict(head=0,rank=16,fit_seed=17,test_seed=23,
              n_fit=65536,n_test_per_measure=65536,length=4,
              corpus_fit_docs=[0,15000],corpus_test_docs=[15000,20000],
              selection="No test-set selection; both direct-fit starts reported",
              results=results)
    output=base/"head0_rank16_fet_cross_n65536.json"
    output.write_text(json.dumps(data,indent=2)+"\n")
    for row in results:print(row)
    return data


if __name__=="__main__":run()

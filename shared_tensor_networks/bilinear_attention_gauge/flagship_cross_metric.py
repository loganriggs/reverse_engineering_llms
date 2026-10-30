"""Evaluate uniform- and corpus-fitted QK1 cuts on fresh, common test measures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from flagship_matrix_free import CORPUS,SNAPSHOT,rows
from matrix_free_environment import SampledBondEnvironment


HERE=Path(__file__).parent
OUTPUT=HERE/"full_width_results"/"cross_metric_seed23.json"


def run():
    torch.set_num_threads(1)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    corpus=np.load(CORPUS,mmap_mode="r")
    records=[]
    for head in (0,4):
        measures={}
        for measure in ("uniform","corpus"):
            _,validation,coverage=rows(state,head,23,1,16384,4,
                distribution=measure,corpus=corpus if measure=="corpus" else None)
            u,v,w,_,_=validation
            measures[measure]=SampledBondEnvironment(u,v,w/w.sum())
        for fit_measure in ("uniform","corpus"):
            suffix="_corpus" if fit_measure=="corpus" else ""
            fit=HERE/"full_width_results"/f"head{head}_seed17_n16384{suffix}.npz"
            matrices=np.load(fit)
            for method in ("wtg","local"):
                for rank in (8,16,32):
                    x=matrices[f"{method}_rank{rank}"]
                    records.append(dict(head=head,fit_measure=fit_measure,
                        test_measure="uniform",method=method,rank=rank,
                        relative_squared_error=measures["uniform"].relative_error(x)))
                    records.append(dict(head=head,fit_measure=fit_measure,
                        test_measure="corpus",method=method,rank=rank,
                        relative_squared_error=measures["corpus"].relative_error(x)))
    result=dict(seed=23,n_test_per_measure=16384,length=4,
        corpus=str(CORPUS),corpus_validation_documents=[15000,20000],
        note="Fit seed 17, independent test seed 23; disjoint corpus documents for fitting and testing",
        records=records)
    OUTPUT.write_text(json.dumps(result,indent=2)+"\n")
    for row in records:
        if row["method"]=="wtg" and row["rank"]==16:
            print(row)
    return result


if __name__=="__main__":
    run()

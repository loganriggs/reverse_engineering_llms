"""Red-team co-activity graph grouping of real rank-96 attention gauges."""

from __future__ import annotations

import json

import numpy as np
import torch

from dual_functional_grouping import connected_groups
from flagship_full_model_eval import get_cuts
from flagship_matrix_free import CORPUS,SNAPSHOT,rows
from flagship_ood_corpora import DATASETS
from flagship_rank96_group_stability import CUTS
from flagship_transfer import ART


THRESHOLDS=(.08,.10,.12,.15)
GAUGES=("random_0","random_1","random_2",
        "intervention_activation_0","intervention_activation_1",
        "intervention_activation_2")


def correlation(part,a,b,rotation):
    u,v,*_=part
    z=np.abs((u@a@rotation)*(v@b@rotation))
    return np.abs(np.corrcoef(z,rowvar=False))


def run(n_selection=16384,n_test=8192):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    fineweb=np.load(CORPUS,mmap_mode="r")
    _,selection,_=rows(state,0,71,1,n_selection,128,
        distribution="corpus",corpus=fineweb,val_docs=(15000,17500))
    tests={}
    for name,path in (("fineweb_test",CORPUS),*DATASETS.items()):
        corpus=np.load(path,mmap_mode="r")
        _,test,_=rows(state,0,73,1,n_test,128,distribution="corpus",
            corpus=corpus,train_docs=(0,600),
            val_docs=(17500,20000) if name=="fineweb_test" else (600,1200))
        tests[name]=test
    del state
    records=[];null_thresholds={}
    for cut_name in CUTS:
        cut=get_cuts()[cut_name]
        left,s,right=np.linalg.svd(cut,full_matrices=False)
        a=left[:,:96]*np.sqrt(s[:96]);b=right[:96].T*np.sqrt(s[:96])
        rotations=np.load(ART/f"head0_exact_gauge_rotations_{cut_name}.npz")
        matrices={gauge:(correlation(selection,a,b,rotations[gauge]),
                         {name:correlation(part,a,b,rotations[gauge])
                          for name,part in tests.items()})
                  for gauge in GAUGES}
        random_edges=np.concatenate([
            matrices[f"random_{seed}"][0][np.triu_indices(96,1)]
            for seed in range(3)])
        null_thresholds[cut_name]=float(np.quantile(random_edges,.999))
        for gauge in GAUGES:
            selection_corr,test_corr=matrices[gauge]
            for threshold in (*THRESHOLDS,null_thresholds[cut_name]):
                edges=np.triu(selection_corr>threshold,1)
                groups=connected_groups(selection_corr,threshold)
                sizes=sorted((len(group) for group in groups if len(group)>1),
                             reverse=True)
                records.append(dict(cut=cut_name,gauge=gauge,
                    threshold=threshold,
                    threshold_source=("random_gauge_99.9_percentile" if
                        threshold==null_thresholds[cut_name] else "fixed_sweep"),
                    n_edges=int(edges.sum()),
                    n_nontrivial_groups=len(sizes),group_sizes=sizes,
                    retained_edge_fractions={name:float(np.mean(
                        test_corr[name][edges]>threshold)) if edges.any() else None
                        for name in tests}))
        print(cut_name,flush=True)
    output=dict(layer=0,head=0,cuts=list(CUTS),gauges=list(GAUGES),
        thresholds=list(THRESHOLDS),n_selection_pairs=n_selection,
        null_thresholds_by_cut=null_thresholds,
        n_test_pairs_per_corpus=n_test,
        selection_fineweb_docs=[15000,17500],
        test_fineweb_docs=[17500,20000],
        graph_rule="edge if absolute Pearson correlation of absolute per-pair component contribution exceeds threshold; connected components",
        scope="threshold sensitivity and held-out edge replication; no semantic labels or full-model group tests",
        records=records)
    (ART/"head0_rank96_coactivity_graph.json").write_text(
        json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":run()

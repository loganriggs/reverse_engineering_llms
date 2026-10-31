"""How many disjoint energy-selected rank-96 groups recur across starts?"""

from __future__ import annotations

import json

import numpy as np
import scipy.optimize as so
import torch

from flagship_full_model_eval import get_cuts
from flagship_matrix_free import CORPUS,SNAPSHOT,rows
from flagship_ood_corpora import DATASETS
from flagship_rank96_group_stability import CUTS,selection_order,weighted_cosine
from flagship_transfer import ART


WIDTH=4
N_GROUPS=12


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
    records=[];assignment_records=[]
    for cut_name in CUTS:
        cut=get_cuts()[cut_name]
        left,s,right=np.linalg.svd(cut,full_matrices=False)
        a=left[:,:96]*np.sqrt(s[:96]);b=right[:96].T*np.sqrt(s[:96])
        identity=np.load(ART/f"head0_exact_gauge_rotations_{cut_name}.npz")
        random_start=np.load(ART/f"head0_exact_gauge_rotations_{cut_name}_random_start.npz")
        sparse={f"i{seed}":identity[f"intervention_activation_{seed}"]
                for seed in range(3)}
        sparse.update({f"r{seed}":random_start[f"intervention_activation_{seed}"]
                       for seed in range(3)})
        control={f"i{seed}":identity[f"random_{seed}"] for seed in range(3)}
        for seed in range(3):
            rng=np.random.default_rng(5500+seed)
            q,r=np.linalg.qr(rng.normal(size=(96,96)))
            control[f"r{seed}"]=q@np.diag(np.sign(np.diag(r)))
        for family,rotations in (("sparse",sparse),("random",control)):
            orders={name:selection_order(selection,a,b,rotation)
                    for name,rotation in rotations.items()}
            bases={name:[rotation[:,orders[name][WIDTH*k:WIDTH*(k+1)]]
                         for k in range(N_GROUPS)]
                   for name,rotation in rotations.items()}
            z={(name,law):(part[0]@a@rotation)*(part[1]@b@rotation)
               for name,rotation in rotations.items()
               for law,part in tests.items()}
            responses={(name,law):[
                z[name,law][:,orders[name][WIDTH*k:WIDTH*(k+1)]].sum(1)
                for k in range(N_GROUPS)]
                for name in rotations for law in tests}
            for seed_a in range(3):
                for seed_b in range(3):
                    ga=f"i{seed_a}";gb=f"r{seed_b}"
                    overlap=np.array([[
                        np.linalg.norm(bases[ga][i].T@bases[gb][j])**2/WIDTH
                        for j in range(N_GROUPS)] for i in range(N_GROUPS)])
                    ia,ib=so.linear_sum_assignment(-overlap)
                    matched=dict(zip(ia.tolist(),ib.tolist()))
                    assignment_records.append(dict(cut=cut_name,family=family,
                        seed_a=seed_a,seed_b=seed_b,
                        matched_block_indices=[matched[i] for i in range(N_GROUPS)],
                        total_assignment_overlap=float(overlap[ia,ib].sum())))
                    for i in range(N_GROUPS):
                        j=matched[i]
                        for law,part in tests.items():
                            records.append(dict(cut=cut_name,family=family,
                                seed_a=seed_a,seed_b=seed_b,group_index=i,
                                matched_index=j,law=law,
                                matched_projector_overlap=float(overlap[i,j]),
                                same_rank_projector_overlap=float(overlap[i,i]),
                                matched_response_cosine=weighted_cosine(
                                    responses[ga,law][i],responses[gb,law][j],part[2])))
        print(cut_name,flush=True)
    output=dict(layer=0,head=0,cuts=list(CUTS),group_width=WIDTH,
        candidate_groups=N_GROUPS,n_selection_pairs=n_selection,
        n_test_pairs_per_corpus=n_test,
        selection_fineweb_docs=[15000,17500],
        test_fineweb_docs=[17500,20000],
        group_rule="disjoint consecutive quartets ranked by unlabeled FineWeb selection pair energy",
        alignment="Hungarian maximum total projector overlap across 12 groups; same alignment used for held-out response cosine",
        random_control="six independent random orthogonal gauges with the same energy selection and assignment",
        scope="number of stable disjoint groups under fixed group width four; not unknown width, semantics, or physical circuit recovery",
        records=records,assignments=assignment_records)
    (ART/"head0_rank96_multiple_groups.json").write_text(
        json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":run()

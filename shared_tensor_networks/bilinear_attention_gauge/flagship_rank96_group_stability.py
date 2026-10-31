"""Do energy-selected groups stabilize when individual sparse axes do not?"""

from __future__ import annotations

import json
from itertools import combinations
import argparse

import numpy as np
import torch

from flagship_full_model_eval import get_cuts
from flagship_matrix_free import CORPUS,SNAPSHOT,rows
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART


CUTS=("uniform_fit_l128_wtg96","corpus_fit_l128_wtg96")
KINDS=("random","activation","intervention_activation",
       "rare_balanced_activation","factor_weight")
SIZES=(4,8,16,32,48,64)


def selection_order(part,a,b,rotation):
    u,v,w,_,_=part
    z=(u@a@rotation)*(v@b@rotation)
    return np.argsort(np.einsum("n,nr->r",w,z*z))[::-1]


def weighted_cosine(x,y,w):
    denominator=np.sqrt(np.dot(w,x*x)*np.dot(w,y*y))
    return float(np.dot(w,x*y)/max(denominator,1e-30))


def run(n_selection=16384,n_test=8192,seeds=(0,1,2),random_start=False):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    fineweb=np.load(CORPUS,mmap_mode="r")
    _,selection,_=rows(state,0,71,1,n_selection,128,
        distribution="corpus",corpus=fineweb,val_docs=(15000,17500))
    test_parts={}
    for law in ("uniform","corpus"):
        _,test,_=rows(state,0,73,1,n_test,128,distribution=law,
            corpus=fineweb if law=="corpus" else None,
            val_docs=(17500,20000))
        test_parts[law]=test
    for name,path in DATASETS.items():
        corpus=np.load(path,mmap_mode="r")
        _,test,_=rows(state,0,73,1,n_test,128,distribution="corpus",
            corpus=corpus,train_docs=(0,600),val_docs=(600,1200))
        test_parts[name]=test
    del state
    records=[];baselines=[]
    for cut_name in CUTS:
        cut=get_cuts()[cut_name]
        left,s,right=np.linalg.svd(cut,full_matrices=False)
        rank=int(np.linalg.matrix_rank(cut))
        assert rank==96
        a=left[:,:rank]*np.sqrt(s[:rank]);b=right[:rank].T*np.sqrt(s[:rank])
        suffix="_random_start" if random_start else ""
        rotations=np.load(ART/f"head0_exact_gauge_rotations_{cut_name}{suffix}.npz")
        full_scores={law:np.einsum("nr,nr->n",part[0]@a,part[1]@b)
                     for law,part in test_parts.items()}
        svd_order=selection_order(selection,a,b,np.eye(rank))
        svd_z={law:(part[0]@a)*(part[1]@b)
               for law,part in test_parts.items()}
        for size in SIZES:
            for law,part in test_parts.items():
                response=svd_z[law][:,svd_order[:size]].sum(1)
                baselines.append(dict(cut=cut_name,size=size,law=law,
                    svd_group_response_energy_fraction=float(
                        np.dot(part[2],response*response)/
                        np.dot(part[2],full_scores[law]**2))))
        kinds=("intervention_activation",) if random_start else KINDS
        for kind in kinds:
            gauges=[f"{kind}_{seed}" for seed in seeds]
            orders={g:selection_order(selection,a,b,rotations[g]) for g in gauges}
            z={(g,law):(part[0]@a@rotations[g])*(part[1]@b@rotations[g])
               for g in gauges for law,part in test_parts.items()}
            for size in SIZES:
                group={g:rotations[g][:,orders[g][:size]] for g in gauges}
                for ga,gb in combinations(gauges,2):
                    overlap=float(np.linalg.norm(group[ga].T@group[gb])**2/size)
                    for law,part in test_parts.items():
                        ra=z[ga,law][:,orders[ga][:size]].sum(1)
                        rb=z[gb,law][:,orders[gb][:size]].sum(1)
                        w=part[2]
                        full=full_scores[law]
                        record=dict(cut=cut_name,kind=kind,size=size,
                            seed_pair=[int(ga.rsplit("_",1)[-1]),
                                       int(gb.rsplit("_",1)[-1])],law=law,
                            projector_overlap=overlap,
                            random_overlap_reference=size/rank,
                            group_response_cosine=weighted_cosine(ra,rb,w),
                            group_a_overlap_with_svd=float(np.linalg.norm(
                                group[ga][svd_order[:size],:])**2/size),
                            group_b_overlap_with_svd=float(np.linalg.norm(
                                group[gb][svd_order[:size],:])**2/size),
                            group_a_cosine_with_svd=weighted_cosine(
                                ra,svd_z[law][:,svd_order[:size]].sum(1),w),
                            group_b_cosine_with_svd=weighted_cosine(
                                rb,svd_z[law][:,svd_order[:size]].sum(1),w),
                            group_a_response_energy_fraction=float(
                                np.dot(w,ra*ra)/np.dot(w,full*full)),
                            group_b_response_energy_fraction=float(
                                np.dot(w,rb*rb)/np.dot(w,full*full)))
                        records.append(record)
        print(cut_name,flush=True)
    output=dict(layer=0,head=0,cuts=list(CUTS),
        kinds=list(("intervention_activation",) if random_start else KINDS),
        random_start=random_start,
        group_sizes=list(SIZES),seeds=list(seeds),
        n_selection_pairs=n_selection,n_test_pairs_per_law=n_test,
        selection_fineweb_docs=[15000,17500],
        test_fineweb_docs=[17500,20000],
        group_rule="top k components by QK2-and-OV-weighted squared pair contribution on FineWeb selection pairs",
        projector_overlap_definition="tr(P_a P_b)/k in rank-96 SVD coordinate space; independent random reference k/96",
        response_cosine_definition="QK2-and-OV-weighted cosine of group QK1 scores on the same held-out pairs",
        scope="stability of a supplied size-k energy-selected group; not discovery of unknown group count or semantic meaning",
        records=records,svd_baselines=baselines)
    suffix="_random_start" if random_start else ""
    (ART/f"head0_rank96_group_stability{suffix}.json").write_text(
        json.dumps(output,indent=2)+"\n")
    return output


def cross_start(n_selection=16384,n_test=8192,seeds=(0,1,2)):
    """Compare groups across independent identity and random start families."""
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
    records=[]
    for cut_name in CUTS:
        cut=get_cuts()[cut_name]
        left,s,right=np.linalg.svd(cut,full_matrices=False)
        rank=96
        a=left[:,:rank]*np.sqrt(s[:rank]);b=right[:rank].T*np.sqrt(s[:rank])
        families={family:np.load(ART/f"head0_exact_gauge_rotations_{cut_name}"
                                      f"{suffix}.npz")
                  for family,suffix in (("identity",""),("random_start","_random_start"))}
        gauges={(family,seed):families[family][f"intervention_activation_{seed}"]
                for family in families for seed in seeds}
        orders={key:selection_order(selection,a,b,rotation)
                for key,rotation in gauges.items()}
        z={(key,name):(part[0]@a@rotation)*(part[1]@b@rotation)
           for key,rotation in gauges.items() for name,part in tests.items()}
        for seed_a in seeds:
            for seed_b in seeds:
                ka=("identity",seed_a);kb=("random_start",seed_b)
                for size in SIZES:
                    pa=gauges[ka][:,orders[ka][:size]]
                    pb=gauges[kb][:,orders[kb][:size]]
                    overlap=float(np.linalg.norm(pa.T@pb)**2/size)
                    for law,part in tests.items():
                        ra=z[ka,law][:,orders[ka][:size]].sum(1)
                        rb=z[kb,law][:,orders[kb][:size]].sum(1)
                        records.append(dict(cut=cut_name,size=size,
                            identity_seed=seed_a,random_start_seed=seed_b,
                            law=law,projector_overlap=overlap,
                            response_cosine=weighted_cosine(ra,rb,part[2])))
        print(cut_name,flush=True)
    output=dict(layer=0,head=0,cuts=list(CUTS),seeds=list(seeds),
        n_selection_pairs=n_selection,n_test_pairs_per_corpus=n_test,
        selection_fineweb_docs=[15000,17500],
        test_fineweb_docs=[17500,20000],
        scope="all 3x3 cross-start pairs of energy-selected intervention-activation groups",
        records=records)
    (ART/"head0_rank96_group_cross_start.json").write_text(
        json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--random-start",action="store_true")
    parser.add_argument("--cross-start",action="store_true")
    args=parser.parse_args()
    cross_start() if args.cross_start else run(random_start=args.random_start)

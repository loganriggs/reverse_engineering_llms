"""How many internal-intervention examples identify sharing in a toy head DAG?

QK and OV factors are known. Detector weights are estimated from noisy
observations. Shared and duplicated graph candidates are observationally
equivalent in the noiseless teacher, but predict different effects when the
first detector node is patched. Select using zero patches and test a held-out
half-scale patch under shifted inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shared_computation_identifiability import attention_factors,forward,parameters


def fit_detectors(q,k,y,p):
    factors=attention_factors(q,k,p)
    design=factors[:,:,None]*k[:,None,:]
    d=k.shape[1]
    shared=np.linalg.lstsq(design.reshape(-1,d),y.reshape(-1),rcond=None)[0]
    duplicate=np.stack([np.linalg.lstsq(design[:,h],y[:,h],rcond=None)[0]
                        for h in range(2)])
    return dict(shared=shared,duplicate=duplicate)


def predict(q,k,p,graph,detectors,patch):
    factors=attention_factors(q,k,p)
    if graph=="shared":
        g=k@detectors["shared"]
        response=np.stack((g,g),axis=1)
        if patch=="zero":response[:]=0
        elif patch=="half":response[:]*=.5
    elif graph=="duplicate":
        response=k@detectors["duplicate"].T
        if patch=="zero":response[:,0]=0
        elif patch=="half":response[:,0]*=.5
    else:raise ValueError(graph)
    return factors*response


def one_trial(rng,teacher,n_obs,n_max_intervention,noise_fraction,d):
    p=parameters(int(rng.integers(0,2**31)),d)
    q=rng.normal(size=(n_obs,d));k=rng.normal(size=(n_obs,d))
    clean=forward(q,k,p,teacher)
    sigma=noise_fraction*float(np.std(clean))
    observed=clean+rng.normal(scale=sigma,size=clean.shape)
    detectors=fit_detectors(q,k,observed,p)
    qi=rng.normal(size=(n_max_intervention,d))
    ki=rng.normal(size=(n_max_intervention,d))
    patched=forward(qi,ki,p,teacher,"first_detector_zero")
    labels=patched+rng.normal(scale=sigma,size=patched.shape)
    candidate={graph:predict(qi,ki,p,graph,detectors,"zero")
               for graph in ("shared","duplicate")}
    # The test changes both the intervention strength and input distribution.
    qt=rng.normal(size=(256,d))*np.geomspace(.4,2.5,d)
    kt=rng.normal(size=(256,d))*np.geomspace(2.5,.4,d)
    target=forward(qt,kt,p,teacher,"first_detector_half")
    test={graph:predict(qt,kt,p,graph,detectors,"half")
          for graph in ("shared","duplicate")}
    result=[]
    for n in (1,2,4,8,16,32):
        losses={graph:float(np.mean((pred[:n]-labels[:n])**2))
                for graph,pred in candidate.items()}
        choice=min(losses,key=losses.get)
        result.append(dict(n_internal=n,correct=choice==teacher,
            selected_test_mse=float(np.mean((test[choice]-target)**2)),
            wrong_test_mse=float(np.mean((test["duplicate" if teacher=="shared" else "shared"]-target)**2))))
    return result


def run(seed=0,trials=100,n_obs=1024,d=6):
    rng=np.random.default_rng(seed)
    cells={}
    for teacher in ("shared","duplicate"):
        for noise in (.1,.5,1.,2.):
            runs=[one_trial(rng,teacher,n_obs,32,noise,d) for _ in range(trials)]
            summaries=[]
            for j,n in enumerate((1,2,4,8,16,32)):
                rows=[r[j] for r in runs]
                summaries.append(dict(n_internal=n,
                    graph_selection_accuracy=float(np.mean([x["correct"] for x in rows])),
                    selected_heldout_half_patch_mse=float(np.mean([x["selected_test_mse"] for x in rows])),
                    wrong_graph_heldout_half_patch_mse=float(np.mean([x["wrong_test_mse"] for x in rows]))))
            cells[f"{teacher}_noise{noise}"]=summaries
    return dict(seed=seed,trials=trials,n_observational=n_obs,d=d,
        observation_only_minimum_parameter_choice="shared",
        observation_only_minimum_parameter_accuracy_by_teacher={"shared":1.,"duplicate":0.},
        supervision="first-detector zero patch on Gaussian inputs",
        heldout="first-detector half-scale patch on shifted-covariance inputs",
        noise_fraction_of_observational_output_std=[.1,.5,1.,2.],
        cells=cells)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--trials",type=int,default=100)
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name(
        "intervention_graph_selection.json"))
    args=parser.parse_args()
    result=run(args.seed,args.trials)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    for key,rows in result["cells"].items():
        print(key,[(r["n_internal"],r["graph_selection_accuracy"]) for r in rows])

"""Eight known graph templates: parameter tying, leaf tying, and state reuse."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def evaluate(bits,x,theta,interventions=()):
    parameter_tie,input_tie,state_reuse=bits
    # Observational inputs: two u coordinates, two left v coordinates,
    # and two right w coordinates. Diagonal observational data have v=w.
    hl=x[:,0]*x[:,1];hr=hl.copy()
    vl=x[:,2].copy();vr=x[:,2].copy() if input_tie else x[:,4].copy()
    zl=x[:,3];zr=x[:,3] if input_tie else x[:,5]
    tl=np.full(len(x),theta[0]);tr=np.full(len(x),theta[0] if parameter_tie else theta[1])
    if 'leaf' in interventions:
        vl+=.5
        if input_tie:vr+=.5
    if 'parameter' in interventions:
        tl[:]=1.1
        if parameter_tie:tr[:]=1.1
    if 'state' in interventions:
        hl=np.full(len(x),1.3)
        if state_reuse:hr=hl.copy()
    return np.column_stack([hl*tl*vl*zl,hr*tr*vr*zr])


def resources(bits):
    p,i,s=bits;mul=7 if s else 8
    return dict(distinct_parameters=1 if p else 2,distinct_input_variables=4 if i else 6,
                multiplication_nodes=mul,arithmetic_operand_edges=2*mul)


def fit_observations(bits,x,y):
    unit=evaluate(bits,x,[1.,1.])
    if bits[0]:
        t=float(np.sum(unit*y)/np.sum(unit*unit));return [t,t]
    return [float(unit[:,j]@y[:,j]/(unit[:,j]@unit[:,j])) for j in range(2)]


def main():
    root=Path(__file__).parent;out=root/'results/instrumented_graph_recovery';out.mkdir(parents=True,exist_ok=False)
    candidates=list(itertools.product([False,True],repeat=3));probes=['parameter','leaf','state']
    protocol=dict(candidate_order=['parameter_tie','input_leaf_tie','intermediate_state_reuse'],candidates=candidates,seeds=list(range(84000,84020)),
                  observations='16 Gaussian points with left/right leaf values equal; both numerical parameters equal',
                  instrumented_probes='32 fresh diagonal Gaussian points per single intervention; known target slot and intervention semantics',
                  evaluation='32 further inputs, diagonal and off-diagonal, with all eight compound intervention subsets',
                  parameter_fitting='least squares using observational labels only; no intervention-driven parameter refit',
                  selection='retain all candidates with max absolute intervention residual <1e-9; complexity only breaks ties',
                  complexity='sum of distinct parameters, distinct input variables, multiplication nodes and operand edges',
                  limitation='known operation/template family and addressed internal access; not blind graph discovery from outputs',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[];max_eval=0.;best_without=0;best_with=0
    for seed in range(84000,84020):
        rng=np.random.default_rng(seed);theta=float(rng.uniform(.5,.95))
        train=rng.normal(size=(16,6));train[:,4:]=train[:,2:4]
        probe=rng.normal(size=(32,6));probe[:,4:]=probe[:,2:4]
        test=rng.normal(size=(32,6))
        for truth in candidates:
            y=evaluate(truth,train,[theta,theta])
            fitted={bits:fit_observations(bits,train,y) for bits in candidates}
            for bits in candidates:np.testing.assert_allclose(evaluate(bits,train,fitted[bits]),y,atol=1e-12,rtol=1e-12)
            for mask in range(8):
                chosen=[name for j,name in enumerate(probes) if mask&(1<<j)]
                possible=[]
                for bits in candidates:
                    error=max([0.]+[float(np.max(np.abs(evaluate(bits,probe,fitted[bits],[p])-evaluate(truth,probe,[theta,theta],[p])))) for p in chosen])
                    if error<1e-9:possible.append(bits)
                expected=2**(3-len(chosen));assert len(possible)==expected
                assert truth in possible
                selected=min(possible,key=lambda b:sum(resources(b).values()))
                rows.append(dict(seed=seed,truth=truth,probe_subset=chosen,consistent_candidates=possible,
                                 complexity_selected=selected,selected_correct=selected==truth))
                if mask==0:best_without+=selected==truth
                if mask==7:
                    assert possible==[truth];best_with+=selected==truth
                    for diagonal in [False,True]:
                        xx=test.copy()
                        if diagonal:xx[:,4:]=xx[:,2:4]
                        for intervention_mask in range(8):
                            interventions=[name for j,name in enumerate(probes) if intervention_mask&(1<<j)]
                            pred=evaluate(selected,xx,fitted[selected],interventions)
                            actual=evaluate(truth,xx,[theta,theta],interventions)
                            err=float(np.max(np.abs(pred-actual)));max_eval=max(max_eval,err);assert err<1e-10
    result=dict(status='passed',teachers=160,subset_records=len(rows),observational_complexity_selected_correct=best_without,
                all_interventions_selected_correct=best_with,max_fresh_compound_intervention_error=max_eval,
                resources=[dict(bits=b,**resources(b)) for b in candidates],records=rows)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()

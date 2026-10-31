"""Exact rank-three gauges for two learned products sharing one variable."""

import json
import numpy as np
from scipy.optimize import linear_sum_assignment,minimize
from scipy.spatial.transform import Rotation

from dual_joint_function_canonical import HERE
from three_count_nonlinear_gauge import TRUTH


def monomial_strengths(matrix,q,dictionary):
    c=(matrix@dictionary.T).reshape(2,3,3)
    h=np.einsum('qic,pij,qjd->pqcd',c,q,c)
    labels=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
    t=np.stack([h[:,:,i,j]*(np.sqrt(2) if i!=j else 1)
                for i,j in labels],axis=-1)
    return t,labels


def profile(matrix,q,dictionary):
    t,labels=monomial_strengths(matrix,q,dictionary)
    energy=np.sum(t*t,axis=(0,1))
    strength=np.sqrt(energy)
    top2=float(np.sort(energy)[-2:].sum()/max(energy.sum(),1e-20))
    top3=float(np.sort(energy)[-3:].sum()/max(energy.sum(),1e-20))
    query_top2=[float(np.sort(np.sum(t[:,query]**2,axis=0))[-2:].sum()/
        max(np.sum(t[:,query]**2),1e-20)) for query in range(2)]
    cross=float(np.sum([energy[k] for k,(i,j) in enumerate(labels) if i!=j])/
                max(energy.sum(),1e-20))
    cosine=np.abs(dictionary@TRUTH.T)
    axis,family=linear_sum_assignment(-cosine)
    mapping={int(f):int(a) for a,f in zip(axis,family)}
    # The rare query rotates backbone logits from A/B/C to B/C/A.
    # Hence its two parity targets use B-C and C-A rather than A-B and B-C.
    edges_by_query=(((0,1),(1,2)),((1,2),(2,0)))
    expected=[];expected_by_query=[]
    for parity in range(2):
        numerator=0.;per_query=[]
        for query in range(2):
            edge=edges_by_query[query][parity]
            i,j=sorted((mapping[edge[0]],mapping[edge[1]]))
            k=labels.index((i,j))
            contribution=float(t[parity,query,k]**2)
            numerator+=contribution
            per_query.append(contribution/max(float(np.sum(t[parity,query]**2)),1e-20))
        expected.append(float(numerator/max(np.sum(t[parity]**2),1e-20)))
        expected_by_query.append(per_query)
    return dict(key_l1=float(np.abs(dictionary).sum()),
        matched_variable_cosine=float(np.mean(cosine[axis,family])),
        matched_family_by_axis=[int(f) for f in family[np.argsort(axis)]],
        product_strengths=strength.tolist(),monomial_labels=[list(x) for x in labels],
        top_two_monomial_energy_fraction=top2,
        top_three_monomial_energy_fraction=top3,
        query_top_two_monomial_energy_fraction=query_top2,
        cross_monomial_energy_fraction=cross,
        intended_pair_energy_fraction=expected,
        intended_pair_fraction_by_output_and_query=expected_by_query)


def fit(matrix,q,base,objective,seed,n_random=256):
    rng=np.random.default_rng(seed)
    starts=rng.uniform(-np.pi,np.pi,size=(n_random,3))
    starts=np.vstack((np.zeros(3),starts))
    def score(x):
        d=Rotation.from_euler('xyz',x).as_matrix()@base
        t,_=monomial_strengths(matrix,q,d)
        energy=np.sum(t*t,axis=(0,1))
        top2=np.sort(energy)[-2:].sum()/max(energy.sum(),1e-20)
        top3=np.sort(energy)[-3:].sum()/max(energy.sum(),1e-20)
        key=np.abs(d).sum()
        return float(objective(key,top2,top3))
    values=np.array([score(x) for x in starts])
    best=(float(values.min()),starts[int(np.argmin(values))])
    for idx in np.argsort(values)[:8]:
        result=minimize(score,starts[idx],method='Powell',
            options={'maxiter':200,'xtol':1e-6,'ftol':1e-8})
        if result.fun<best[0]:best=(float(result.fun),result.x)
    return Rotation.from_euler('xyz',best[1]).as_matrix()@base,best[0],values


def run():
    analysis=json.loads((HERE/'shared_pair_count_analysis.json').read_text())
    rows=[]
    for row in analysis['rows']:
        seed=row['seed'];q=np.asarray(row['quadratic_effective'])
        for reduction in ('uniform_rank3','empirical_rank3'):
            matrix=np.asarray(row['matrices'][reduction])
            _,s,v=np.linalg.svd(matrix,full_matrices=False)
            assert s[3]<1e-7
            base=v[:3]
            sparse,_,random_values=fit(matrix,q,base,
                lambda key,top2,top3:key,610000+seed)
            key_scale=np.abs(sparse).sum()
            choices={'svd':base,'key_l1':sparse}
            for lam in (.3,1,3):
                d,_,_=fit(matrix,q,base,
                    lambda key,top2,top3:key/key_scale+lam*(1-top2),
                    620000+100*seed+int(lam*10))
                choices[f'graph2_lambda{lam}']=d
            for lam in (1,3):
                d,_,_=fit(matrix,q,base,
                    lambda key,top2,top3:key/key_scale+lam*(1-top3),
                    640000+100*seed+int(lam*10))
                choices[f'graph3_lambda{lam}']=d
            choices_out={}
            for name,d in choices.items():
                assert np.max(np.abs(d.T@d-base.T@base))<1e-9
                choices_out[name]=profile(matrix,q,d)
                choices_out[name]['dictionary']=d.tolist()
            rng=np.random.default_rng(630000+seed)
            controls=[profile(matrix,q,Rotation.random(random_state=rng).as_matrix()@base)
                      for _ in range(256)]
            rows.append(dict(seed=seed,reduction=reduction,
                cut_task_scores=row['scores'][reduction],
                gauges=choices_out,
                random_control={field:float(np.mean([c[field] for c in controls]))
                    for field in ('key_l1','matched_variable_cosine',
                        'top_two_monomial_energy_fraction',
                        'top_three_monomial_energy_fraction',
                        'cross_monomial_energy_fraction')},
                random_control_n=len(controls)))
        print(seed,flush=True)
    result=dict(source='shared_pair_count_analysis.json',
        gauge_fit_uses_task_labels=False,rank=3,
        description='two parity outputs share B; exact orthogonal gauge optimization',
        rows=rows)
    (HERE/'shared_pair_gauge.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()

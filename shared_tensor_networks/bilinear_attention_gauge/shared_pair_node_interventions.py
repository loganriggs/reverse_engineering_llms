"""Query/output response maps for deleting exact-gauge quadratic nodes."""

import json
import numpy as np

from dual_joint_function_canonical import HERE
from shared_pair_gauge import monomial_strengths
from three_count_task_fit import count_features
from train_three_count_language import examples


EXPECTED={(0,1):((0,0),), (1,2):((0,1),(1,0)), (0,2):((1,1),)}


def run(analysis_name='shared_pair_count_analysis.json',
        gauge_name='shared_pair_gauge.json',example_source=examples,
        output_name='shared_pair_node_interventions.json'):
    analysis=json.loads((HERE/analysis_name).read_text())
    gauges=json.loads((HERE/gauge_name).read_text())
    rows=[]
    for row in gauges['rows']:
        seed=row['seed'];reduction=row['reduction']
        source=next(x for x in analysis['rows'] if x['seed']==seed)
        matrix=np.asarray(source['matrices'][reduction])
        q=np.asarray(source['quadratic_effective'])
        tokens,_,_=example_source(30000,530000+seed,False)
        tokens=tokens.numpy()
        counts=count_features(tokens).astype(np.float64)
        query=(tokens[:,-1]-8).astype(int)
        choices={}
        for name in ('svd','key_l1','joint_nullspace','graph3_lambda3',
                     'cell_one_only','cell_one_lambda1'):
            gauge=row['gauges'][name]
            d=np.asarray(gauge['dictionary'])
            t,labels=monomial_strengths(matrix,q,d)
            z=counts@d.T
            basis=np.stack((z[:,0]**2,np.sqrt(2)*z[:,0]*z[:,1],
                np.sqrt(2)*z[:,0]*z[:,2],z[:,1]**2,
                np.sqrt(2)*z[:,1]*z[:,2],z[:,2]**2),axis=1)
            families=gauge['matched_family_by_axis']
            mapping={int(f):axis for axis,f in enumerate(families)}
            edge_maps={}
            for edge,cells in EXPECTED.items():
                i,j=sorted((mapping[edge[0]],mapping[edge[1]]))
                node=labels.index((i,j))
                energy=np.zeros((2,2))
                for query_id in range(2):
                    mask=query==query_id
                    for parity in range(2):
                        effect=t[parity,query_id,node]*basis[mask,node]
                        energy[query_id,parity]=np.mean(effect**2)
                expected=sum(energy[query_id,parity] for query_id,parity in cells)
                selectivity=float(expected/max(energy.sum(),1e-20))
                edge_maps[f'{edge[0]}-{edge[1]}']=dict(node=labels[node],
                    response_energy_by_query_and_output=energy.tolist(),
                    expected_cell_energy_fraction=selectivity)
            choices[name]=edge_maps
        rows.append(dict(seed=seed,reduction=reduction,gauges=choices))
        print(seed,reduction,flush=True)
    result=dict(source=gauge_name,
        expected_cells={'A-B':'common query, first parity',
                        'B-C':'common query second parity and rare query first parity',
                        'A-C':'rare query second parity'},
        labels_used_in_gauge_fit=False,
        note='Known labels enter only post-fit axis matching and selectivity scoring',
        rows=rows)
    (HERE/output_name).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()

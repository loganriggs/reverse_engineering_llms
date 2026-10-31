"""Test whether three shared quadratic nodes retain the new task."""

import json
import numpy as np
import torch

from dual_joint_function_canonical import HERE
from shared_pair_gauge import monomial_strengths
from three_count_task_fit import count_features
from train_shared_pair_count import SharedPairCountModel,targets
from train_three_count_language import examples


def monomial_features(z):
    return np.stack((z[:,0]**2,np.sqrt(2)*z[:,0]*z[:,1],
                     np.sqrt(2)*z[:,0]*z[:,2],z[:,1]**2,
                     np.sqrt(2)*z[:,1]*z[:,2],z[:,2]**2),axis=1)


def run(analysis_name='shared_pair_count_analysis.json',
        gauge_name='shared_pair_gauge.json',example_source=examples,
        model_prefix='shared_pair_count',output_name='shared_pair_graph_prune.json'):
    analysis=json.loads((HERE/analysis_name).read_text())
    gauges=json.loads((HERE/gauge_name).read_text())
    rows=[]
    for row in gauges['rows']:
        seed=row['seed'];reduction=row['reduction']
        source=next(x for x in analysis['rows'] if x['seed']==seed)
        matrix=np.asarray(source['matrices'][reduction])
        q=np.asarray(source['quadratic_effective'])
        model=SharedPairCountModel().double()
        model.load_state_dict(torch.load(HERE/f'{model_prefix}_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        bias=model.backbone.read.bias.detach().numpy()
        pb=model.parity_bias.detach().numpy()
        tokens,bits,attrs=example_source(30000,530000+seed,False)
        tokens=tokens.numpy();truth=targets(bits).numpy().astype(bool)
        counts=count_features(tokens).astype(np.float64)
        query=(tokens[:,-1]-8).astype(int)
        logits=bias+np.einsum('nk,nok->no',counts,matrix.reshape(2,3,8)[query])
        parity=np.einsum('ni,pij,nj->np',logits,q,logits)+pb
        native=np.c_[logits,parity]
        masks={'all':np.ones(len(tokens),dtype=bool),'rare_heldout':attrs['heldout']}
        choices={}
        for name in ('svd','key_l1','joint_nullspace','graph2_lambda3','graph3_lambda3',
                     'cell_one_only','cell_one_lambda1'):
            d=np.asarray(row['gauges'][name]['dictionary'])
            t,labels=monomial_strengths(matrix,q,d)
            energy=np.sum(t*t,axis=(0,1))
            keep=np.argsort(energy)[-3:]
            omit=np.array([i for i in range(6) if i not in keep])
            z=counts@d.T
            features=monomial_features(z)
            quadratic_sum=np.einsum('npk,nk->np',t[:,query,:].transpose(1,0,2),features)
            signal=logits-bias
            direct_quadratic=np.einsum('ni,pij,nj->np',signal,q,signal)
            assert np.max(np.abs(quadratic_sum[:256]-direct_quadratic[:256]))<1e-8
            removed=np.einsum('npk,nk->np',t[:,query,:].transpose(1,0,2)[:,:,omit],
                              features[:,omit])
            output=np.c_[logits,parity-removed]
            choices[name]=dict(kept_nodes=[labels[i] for i in keep],
                removed_quadratic_energy_fraction=float(energy[omit].sum()/energy.sum()),
                relative_parity_output_mse=float(np.sum(removed**2)/
                    np.maximum(np.sum(parity**2),1e-20)),
                scores={group:dict(five_output_accuracy=float(np.mean(np.all((output[mask]>0)==truth[mask],axis=1))),
                    pair_joint_accuracy=float(np.mean(np.all((output[mask,3:]>0)==truth[mask,3:],axis=1))),
                    pair_accuracy=[float(np.mean((output[mask,3+j]>0)==truth[mask,3+j]))
                                   for j in range(2)]) for group,mask in masks.items()})
        rows.append(dict(seed=seed,reduction=reduction,
            unpruned={group:dict(five_output_accuracy=float(np.mean(np.all((native[mask]>0)==truth[mask],axis=1))),
                pair_joint_accuracy=float(np.mean(np.all((native[mask,3:]>0)==truth[mask,3:],axis=1))))
                for group,mask in masks.items()},gauges=choices))
        print(seed,reduction,flush=True)
    result=dict(source=gauge_name,
        selection='three largest group-squared quadratic coefficients across both outputs and queries',
        selection_uses_task_labels=False,test='30000 independent sentences and rare whole-combination holdout',
        rows=rows)
    (HERE/output_name).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()

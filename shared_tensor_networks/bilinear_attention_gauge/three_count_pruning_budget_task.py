"""Evaluate fixed post-pruning gauges on independent task sentences."""

import json
import numpy as np
import torch

from analyze_three_count_language import flatten,head_tables
from dual_joint_function_canonical import HERE
from three_count_graph_prior import terms
from three_count_metric import pair_weights,weighted_fit
from three_count_nonlinear_gauge import rotation
from three_count_task_fit import count_features
from train_three_count_composed import ComposedCountModel
from train_three_count_language import examples


def run():
    source=json.loads((HERE/'three_count_pruning_budget.json').read_text())
    rows=[]
    for seed in source['seeds']:
        model=ComposedCountModel().double()
        model.load_state_dict(torch.load(HERE/f'three_count_composed_scratch_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        matrix=flatten(sum(head_tables(model.backbone)))
        scale=model.logit_scale.detach().numpy()
        q=model.quadratic.detach().numpy()
        q=.5*(q+q.T)/scale[:,None]/scale[None,:]
        bias=model.backbone.read.bias.detach().numpy()
        pb=float(model.parity_bias.detach())
        train,_,_=examples(20000,110000+seed,True)
        u,s,v=np.linalg.svd(matrix,full_matrices=False)
        cuts={'uniform_rank2':(u[:,:2]*s[:2])@v[:2]}
        cuts['empirical_rank2'],_=weighted_fit(matrix,pair_weights(train.numpy()),2)
        test,bits,attrs=examples(30000,130000+seed,False)
        tokens=test.numpy();bits=bits.numpy().astype(bool)
        counts=count_features(tokens).astype(np.float64)
        query=(tokens[:,-1]-8).astype(int)
        truth=np.c_[bits,np.logical_xor(bits[:,0],bits[:,1])]
        masks={'all':np.ones(len(tokens),dtype=bool),'rare_heldout':attrs['heldout']}
        for cut,m in cuts.items():
            _,sv,base=np.linalg.svd(m,full_matrices=False)
            assert sv[2]<1e-7
            logits=bias+np.einsum('nk,nok->no',counts,m.reshape(2,3,8)[query])
            parity=np.einsum('ni,ij,nj->n',logits,q,logits)+pb
            original=np.c_[logits,parity]>0
            selected=next(r['selections'] for r in source['rows']
                          if r['seed']==seed and r['reduction']==cut)
            methods={}
            for name,choice in selected.items():
                d=rotation(choice['angle'])@base[:2]
                node=choice['deleted_node']
                z=counts@d.T
                basis=np.stack((z[:,0]**2,np.sqrt(2)*z[:,0]*z[:,1],z[:,1]**2),axis=1)
                deleted=terms(m,q,d)[query][:,node]*basis[:,node]
                pred=np.c_[logits,parity-deleted]>0
                methods[name]={group:dict(
                    parity_accuracy=float(np.mean(pred[mask,3]==truth[mask,3])),
                    four_output_accuracy=float(np.mean(np.all(pred[mask]==truth[mask],axis=1))))
                    for group,mask in masks.items()}
            rows.append(dict(seed=seed,reduction=cut,methods=methods,
                unpruned={group:dict(parity_accuracy=float(np.mean(original[mask,3]==truth[mask,3])),
                    four_output_accuracy=float(np.mean(np.all(original[mask]==truth[mask],axis=1))))
                    for group,mask in masks.items()}))
        print(seed,flush=True)
    result=dict(source='three_count_pruning_budget.json',test='30000 independent sentences per seed',
                task_labels_used_in_gauge_fit=False,rows=rows)
    (HERE/'three_count_pruning_budget_task.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()

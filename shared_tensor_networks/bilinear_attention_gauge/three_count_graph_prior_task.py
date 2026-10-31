"""Held-out task test after deleting one graph node chosen without labels."""

import json
import numpy as np
import torch

from dual_joint_function_canonical import HERE
from three_count_graph_prior import terms
from three_count_nonlinear_gauge import edit_panel, rotation
from three_count_task_fit import count_features
from train_three_count_composed import ComposedCountModel
from train_three_count_language import examples


def run():
    torch.set_num_threads(1)
    source=json.loads((HERE/'three_count_graph_prior.json').read_text())
    analysis=json.loads((HERE/'three_count_composed_scratch_analysis.json').read_text())
    lookup={r['seed']:r for r in analysis['rows']}
    rows=[]
    for row in source['rows']:
        seed=row['seed'];reduction=row['reduction']
        matrix=np.asarray(lookup[seed]['rank2_matrices'][reduction])
        q=np.asarray(lookup[seed]['symmetric_quadratic_effective'])
        _,_,v=np.linalg.svd(matrix,full_matrices=False)
        model=ComposedCountModel().double()
        model.load_state_dict(torch.load(HERE/f'three_count_composed_scratch_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        bias=model.backbone.read.bias.detach().numpy()
        pb=float(model.parity_bias.detach())
        tokens,bits,attrs=examples(30000,130000+seed,False)
        tokens=tokens.numpy();bits=bits.numpy().astype(bool)
        counts=count_features(tokens).astype(np.float64)
        query=(tokens[:,-1]-8).astype(int)
        truth=np.c_[bits,np.logical_xor(bits[:,0],bits[:,1])]
        reduced_bits=bias+np.einsum('nk,nok->no',counts,
                                      matrix.reshape(2,3,8)[query])
        parity=np.einsum('ni,ij,nj->n',reduced_bits,q,reduced_bits)+pb
        original=np.c_[reduced_bits,parity]>0
        choices={**row['baselines'],
                 'graph_tau0.1_lambda0.3':row['candidates']['tau0.1_lambda0.3']}
        methods={}
        for name,choice in choices.items():
            d=rotation(choice['angle'])@v[:2]
            z=counts@d.T
            coef=terms(matrix,q,d)[query]
            features=np.stack((z[:,0]**2,np.sqrt(2)*z[:,0]*z[:,1],z[:,1]**2),axis=1)
            node=choice['pruning']['deleted_node']
            pruned=np.c_[reduced_bits,parity-coef[:,node]*features[:,node]]>0
            methods[name]={}
            for group,mask in (('all',np.ones(len(tokens),dtype=bool)),
                               ('rare_heldout',attrs['heldout'])):
                methods[name][group]=dict(parity_accuracy=float(np.mean(pruned[mask,3]==truth[mask,3])),
                    all_four_accuracy=float(np.mean(np.all(pruned[mask]==truth[mask],axis=1))))
        rows.append(dict(seed=seed,reduction=reduction,
            original={group:dict(parity_accuracy=float(np.mean(original[mask,3]==truth[mask,3])),
                                 all_four_accuracy=float(np.mean(np.all(original[mask]==truth[mask],axis=1))))
                for group,mask in (('all',np.ones(len(tokens),dtype=bool)),
                                   ('rare_heldout',attrs['heldout']))},methods=methods))
        print(seed,reduction,flush=True)
    result=dict(source='three_count_graph_prior.json',task_labels_used_in_fitting=False,
                test='independent 30000 sentences per seed with rare whole-combination holdout',rows=rows)
    (HERE/'three_count_graph_prior_task.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()

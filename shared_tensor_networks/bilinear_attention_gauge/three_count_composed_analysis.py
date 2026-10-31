"""Exact token-table and rank controls for trained bilinear compositions."""

import json

import numpy as np
import torch

from analyze_three_count_language import flatten,head_tables
from dual_joint_function_canonical import HERE
from three_count_metric import pair_weights,weighted_fit
from three_count_nonlinear_metric import forward
from three_count_task_fit import count_features
from train_three_count_composed import ComposedCountModel
from train_three_count_language import examples


def readout(bits,quadratic,bias):
    return np.einsum('ni,ij,nj->n',bits,quadratic,bits)+bias


def score(matrix,bias,quadratic,parity_bias,tokens,target,attrs,reference):
    panel=dict(counts=count_features(tokens).astype(np.float64),
               query=(tokens[:,-1]-8).astype(np.int64))
    bits=forward(matrix,bias,panel)
    parity=readout(bits,quadratic,parity_bias)
    output=np.c_[bits,parity]
    truth=np.c_[target,np.logical_xor(target[:,0],target[:,1])]
    masks={'all':np.ones(len(tokens),dtype=bool),
           'rare_heldout':attrs['heldout']}
    result={}
    for name,mask in masks.items():
        result[name]=dict(n=int(mask.sum()),
            parity_accuracy=float(np.mean((parity[mask]>0)==truth[mask,3])),
            bit_joint_accuracy=float(np.mean(np.all((bits[mask]>0)==truth[mask,:3],axis=1))),
            all_four_accuracy=float(np.mean(np.all((output[mask]>0)==truth[mask],axis=1))))
    if reference is not None:
        result['relative_output_mse']=float(np.sum((output-reference)**2)/
                                            np.sum(reference**2))
    return result,output


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(10):
        train,_,_=examples(20000,61000+seed,True)
        test,targets,attrs=examples(30000,30000+seed,False)
        train_np=train.numpy();test_np=test.numpy()
        target=targets.numpy().astype(bool)
        weight=pair_weights(train_np)
        for stage,path in (
            ('frozen',HERE/f'three_count_composed_frozen_seed{seed}.pt'),
            ('joint',HERE/f'three_count_composed_seed{seed}.pt')):
            model=ComposedCountModel().double()
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            model.eval()
            matrix=flatten(sum(head_tables(model.backbone)))
            bias=model.backbone.read.bias.detach().numpy()
            scale=model.logit_scale.detach().numpy()
            q=model.quadratic.detach().numpy()
            q=.5*(q+q.T)/scale[:,None]/scale[None,:]
            pb=float(model.parity_bias.detach())
            u,s,v=np.linalg.svd(matrix,full_matrices=False)
            candidates={'native':matrix,
                'uniform_rank2':(u[:,:2]*s[:2])@v[:2],
                'uniform_rank3':(u[:,:3]*s[:3])@v[:3]}
            candidates['empirical_rank2'],_=weighted_fit(matrix,weight,2)
            with torch.no_grad():
                direct=model(test[:256]).detach().numpy()
            _,proxy=score(matrix,bias,q,pb,test_np[:256],target[:256],
                          {k:v[:256] for k,v in attrs.items()},direct)
            gate=float(np.max(np.abs(direct-proxy)))
            assert gate<1e-9,(seed,stage,gate)
            native=score(matrix,bias,q,pb,test_np,target,attrs,None)[1]
            scores={}
            for name,m in candidates.items():
                row,_=score(m,bias,q,pb,test_np,target,attrs,native)
                row['uniform_table_mse']=float(np.sum((m-matrix)**2)/
                                               np.sum(matrix**2))
                scores[name]=row
            rows.append(dict(seed=seed,stage=stage,exact_forward_gate=gate,
                singular_values=s.tolist(),symmetric_quadratic_effective=q.tolist(),
                scores=scores,
                rank2_matrices={k:v.tolist() for k,v in candidates.items()
                                if k.endswith('rank2')}))
            print(seed,stage,scores['native']['rare_heldout']['all_four_accuracy'],
                  scores['uniform_rank2']['all']['parity_accuracy'],
                  scores['uniform_rank3']['all']['all_four_accuracy'],flush=True)
    result=dict(n_models=10,stages=['frozen','joint'],
        exact_object='first-layer query/output-by-key 6x8 table followed by learned quadratic readout',
        fit_labels_used=False,rank2_fit_train='20,000 heldout-free unlabeled training-law contexts',
        test='30,000 independent contexts per model, including whole-combination holdout',
        rows=rows)
    (HERE/'three_count_composed_analysis.json').write_text(
        json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()

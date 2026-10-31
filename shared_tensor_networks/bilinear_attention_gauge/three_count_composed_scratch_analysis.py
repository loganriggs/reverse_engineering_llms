"""Exact-function and rank analysis of compositions trained from scratch."""

import json

import numpy as np
import torch

from analyze_three_count_language import flatten,head_tables
from dual_joint_function_canonical import HERE
from three_count_composed_analysis import score
from three_count_metric import pair_weights,weighted_fit
from train_three_count_composed import ComposedCountModel
from train_three_count_language import examples


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(10):
        train,_,_=examples(20000,110000+seed,True)
        test,targets,attrs=examples(30000,130000+seed,False)
        test_np=test.numpy();target=targets.numpy().astype(bool)
        model=ComposedCountModel().double()
        model.load_state_dict(torch.load(
            HERE/f'three_count_composed_scratch_seed{seed}.pt',
            map_location='cpu',weights_only=True))
        model.eval()
        matrix=flatten(sum(head_tables(model.backbone)))
        bias=model.backbone.read.bias.detach().numpy()
        scale=model.logit_scale.detach().numpy()
        q=model.quadratic.detach().numpy()
        q=.5*(q+q.T)/scale[:,None]/scale[None,:]
        pb=float(model.parity_bias.detach())
        with torch.no_grad():direct=model(test[:256]).detach().numpy()
        _,proxy=score(matrix,bias,q,pb,test_np[:256],target[:256],
                      {k:v[:256] for k,v in attrs.items()},direct)
        gate=float(np.max(np.abs(direct-proxy)))
        assert gate<1e-9,gate
        native=score(matrix,bias,q,pb,test_np,target,attrs,None)[1]
        u,s,v=np.linalg.svd(matrix,full_matrices=False)
        candidates={'native':matrix,
            'uniform_rank2':(u[:,:2]*s[:2])@v[:2],
            'uniform_rank3':(u[:,:3]*s[:3])@v[:3]}
        candidates['empirical_rank2'],_=weighted_fit(
            matrix,pair_weights(train.numpy()),2)
        scores={}
        for name,m in candidates.items():
            result,_=score(m,bias,q,pb,test_np,target,attrs,native)
            result['uniform_table_mse']=float(np.sum((m-matrix)**2)/
                                              np.sum(matrix**2))
            scores[name]=result
        q_share=float((abs(q[0,1])+abs(q[1,0]))/np.abs(q).sum())
        rows.append(dict(seed=seed,exact_forward_gate=gate,
            singular_values=s.tolist(),
            symmetric_quadratic_effective=q.tolist(),
            first_two_logit_interaction_weight_share=q_share,
            scores=scores,
            rank2_matrices={k:m.tolist() for k,m in candidates.items()
                            if k.endswith('rank2')}))
        print(seed,scores['native']['all']['all_four_accuracy'],
              scores['empirical_rank2']['all']['parity_accuracy'],
              scores['uniform_rank3']['rare_heldout']['all_four_accuracy'],flush=True)
    result=dict(n_models=10,initialization='all model weights random',
        test='independent 30,000-sentence whole-combination holdout per model',
        train_measure='20,000 unlabeled training-law sentences per model',
        exact_object='6x8 query/output-by-key attention table and trained quadratic readout',
        fit_labels_used=False,rows=rows)
    (HERE/'three_count_composed_scratch_analysis.json').write_text(
        json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()

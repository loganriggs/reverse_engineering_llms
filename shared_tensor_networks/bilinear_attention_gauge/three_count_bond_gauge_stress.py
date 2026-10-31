"""Exact bond-gauge rescaling versus function-level rank selection."""

import json

import numpy as np
import torch

from analyze_three_count_language import flatten,head_tables
from dual_joint_function_canonical import HERE
from three_count_rank_selection import WITHIN,response_error,task_scores
from train_three_count_composed import ComposedCountModel
from train_three_count_language import examples


EPSILONS=(1.,.3,.1,.03)
RANKS=(1,2,3,4,5,6)
THRESHOLDS=(.01,.05)


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(10):
        model=ComposedCountModel().double()
        model.load_state_dict(torch.load(HERE/f'three_count_composed_scratch_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        table=flatten(sum(head_tables(model.backbone)))
        u,s,v=np.linalg.svd(table,full_matrices=False)
        left=u*np.sqrt(s)[None,:]
        right=np.sqrt(s)[:,None]*v
        assert np.max(np.abs(left@right-table))<1e-12
        scale=model.logit_scale.detach().numpy()
        q=model.quadratic.detach().numpy()
        q=.5*(q+q.T)/scale[:,None]/scale[None,:]
        bias=model.backbone.read.bias.detach().numpy()
        pb=float(model.parity_bias.detach())
        test,bits,attrs=examples(30000,130000+seed,False)
        test=test.numpy();bits=bits.numpy()
        for epsilon in EPSILONS:
            diagonal=np.array([1.,1.,epsilon,1.,1.,1.])
            occurrence_right=diagonal[:,None]*right
            occurrence_left=left/diagonal[None,:]
            assert np.max(np.abs(occurrence_left@occurrence_right-table))<1e-12
            ub,sb,vb=np.linalg.svd(occurrence_right,full_matrices=False)
            for rank in RANKS:
                hidden_cut=(ub[:,:rank]*sb[:rank])@vb[:rank]
                hidden_decoded=occurrence_left@hidden_cut
                function_cut=(u[:,:rank]*s[:rank])@v[:rank]
                for method,cut in [('internal_bond',hidden_decoded),
                                   ('full_function',function_cut)]:
                    hidden_error=(float(np.sum((hidden_cut-occurrence_right)**2)/
                                         np.sum(occurrence_right**2)) if
                                  method=='internal_bond' else None)
                    function_error=float(np.sum((cut-table)**2)/np.sum(table**2))
                    edit=max(response_error(table,cut,WITHIN,query)
                             for query in [0,1])
                    rows.append(dict(seed=seed,epsilon=epsilon,method=method,
                        rank=rank,hidden_relative_mse=hidden_error,
                        full_function_relative_mse=function_error,
                        worst_query_edit_relative_mse=edit,
                        task=task_scores(cut,q,bias,pb,test,bits,attrs)))
        print('seed',seed,flush=True)
    selections=[]
    for seed in range(10):
        for epsilon in EPSILONS:
            for method in ('internal_bond','full_function'):
                pool=[r for r in rows if r['seed']==seed and r['epsilon']==epsilon
                      and r['method']==method]
                for threshold in THRESHOLDS:
                    criteria=('hidden_energy','decoded_edit') if method=='internal_bond' else ('full_function_energy','decoded_edit')
                    for criterion in criteria:
                        field={'hidden_energy':'hidden_relative_mse',
                               'full_function_energy':'full_function_relative_mse',
                               'decoded_edit':'worst_query_edit_relative_mse'}[criterion]
                        chosen=next((r for r in pool if r[field]<=threshold),pool[-1])
                        selections.append(dict(seed=seed,epsilon=epsilon,
                            method=method,criterion=criterion,threshold=threshold,
                            selected_rank=chosen['rank'],
                            full_function_relative_mse=chosen['full_function_relative_mse'],
                            rare_heldout_accuracy=chosen['task']['rare_heldout']['all_four_accuracy'],
                            all_accuracy=chosen['task']['all']['all_four_accuracy']))
    result=dict(description='Exact GL bond diagonal rescaling on the SVD factorization of ten trained 6x8 toy-language attention functions; compare internal-factor energy, full-function SVD, and answer-free decoded token-edit fidelity for rank choice',
        gauge='A -> A diag(1,1,1/epsilon,1,1,1); B -> diag(1,1,epsilon,1,1,1) B',
        exact_function_invariance_checked=True,
        epsilons=EPSILONS,ranks=RANKS,thresholds=THRESHOLDS,
        rank_selection_uses_task_labels=False,task_labels_used_only_for_evaluation=True,
        rows=rows,selections=selections)
    (HERE/'three_count_bond_gauge_stress.json').write_text(json.dumps(result,indent=2)+'\n')
    for epsilon in EPSILONS:
        for method,criterion in [('internal_bond','hidden_energy'),
                                 ('internal_bond','decoded_edit'),
                                 ('full_function','full_function_energy')]:
            subset=[x for x in selections if x['epsilon']==epsilon and
                    x['method']==method and x['criterion']==criterion and
                    x['threshold']==.05]
            print(epsilon,method,criterion,'ranks',
                  [x['selected_rank'] for x in subset],
                  'rare',round(float(np.mean([x['rare_heldout_accuracy']
                                              for x in subset])),4),flush=True)


if __name__=='__main__':run()

"""Independent-seed confirmation of two preselected causal priors.

Seeds 5–9 were not used for the exploratory beta/gamma sweep. The two
weights here are fixed: output beta=.3 and finite-edit gamma=.1.
"""

import json

import numpy as np
import torch

from contextual_composition_causal_prior import (
    PhysicalTopK, evaluate, make_panel, train_one)
from contextual_composition_reduction import FIT_QUERIES, exact_head, head_table
from contextual_composition_shared_groups import FlatSparse, train_featurizer
from dual_joint_function_canonical import HERE
from train_contextual_composition_attention import (
    ContextualCompositionAttention, FAMILIES, examples)


CANDIDATES={'reconstruction_baseline':(0.,0.),
            'output_beta0.3':(.3,0.),
            'edit_gamma0.1':(0.,.1)}


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(5,10):
        native=ContextualCompositionAttention()
        native.load_state_dict(torch.load(HERE/f'contextual_composition_attention_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        native.eval();native.requires_grad_(False)
        table=head_table(native)
        fit_x,_,fit_q=examples(40000,20262400+seed)
        val_x,_,val_q=examples(10000,20262500+seed)
        fit_h=torch.from_numpy(exact_head(table,fit_x[
            np.isin(fit_q,FIT_QUERIES)])).float()
        val_h=torch.from_numpy(exact_head(table,val_x[
            np.isin(val_q,FIT_QUERIES)])).float()
        torch.manual_seed(20262800+seed)
        flat=FlatSparse(12)
        flat_fit=train_featurizer(flat,fit_h,val_h,1200,20262800+seed*100)
        baseline_state={k:v.detach().clone() for k,v in flat.state_dict().items()}
        torch.save(baseline_state,
                   HERE/f'contextual_composition_groups_seed{seed}_flat_12.pt')

        train=make_panel(native,table,30000,20263500+seed,fit_only=True)
        val=make_panel(native,table,8000,20263600+seed,fit_only=True)
        test=make_panel(native,table,20000,20263700+seed)
        shifted=make_panel(native,table,20000,20263800+seed,shifted=True)
        methods={}
        for index,(name,(beta,gamma)) in enumerate(CANDIDATES.items()):
            model=FlatSparse(12)
            model.load_state_dict(baseline_state)
            if name=='reconstruction_baseline':
                fit=dict(selected_step=0,improved=False)
            else:
                # Same fixed seed formula as the exploratory experiment;
                # candidate weights were frozen before these model seeds.
                original_index={'output_beta0.3':2,'edit_gamma0.1':4}[name]
                fit=train_one(model,native,train,val,beta,gamma,
                              20263900+seed*100+original_index)
                torch.save(model.state_dict(),
                           HERE/f'contextual_composition_causal_seed{seed}_{name}.pt')
            methods[name]=dict(fit=fit,test=evaluate(model,native,test),
                               shifted=evaluate(model,native,shifted))
        for k in (3,4):
            name=f'physical_top{k}'
            model=PhysicalTopK(k)
            methods[name]=dict(fit=dict(selected_step=None),
                               test=evaluate(model,native,test),
                               shifted=evaluate(model,native,shifted))
        rows.append(dict(seed=seed,flat_fit=flat_fit,methods=methods))
        print('seed',seed,'confirmed',
              {name:round(method['test']['xor']['edit_path_relative_mse'],4)
               for name,method in methods.items()},flush=True)
    summary={}
    for name in (*CANDIDATES,'physical_top3','physical_top4'):
        summary[name]={}
        for split in ('test','shifted'):
            summary[name][split]={family:{metric:float(np.mean([
                row['methods'][name][split][family][metric] for row in rows]))
                for metric in ('accuracy','head_relative_mse','relative_logit_mse',
                               'edit_path_relative_mse','edit_sign_agreement')}
                for family in FAMILIES}
        summary[name]['improved_seeds']=sum(row['methods'][name]['fit'].get('improved',False)
                                             for row in rows)
    record=dict(description='Independent model seeds 5–9, trained on all 16 tasks. Flat 12-atom featurizer fitted only on primitive+AND queries. Beta=.3 output fidelity and gamma=.1 finite-edit fidelity were fixed from exploratory seeds 0–4 before seeing these seeds. Same <=5% seen-query validation reconstruction gate; XOR entirely absent from featurizer fitting and checkpoint selection. Uniform and shifted-token tests, physical top-three/top-four controls on identical examples.',
                fixed_candidates=CANDIDATES,rows=rows,summary=summary)
    (HERE/'contextual_composition_causal_confirmation.json').write_text(
        json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':run()

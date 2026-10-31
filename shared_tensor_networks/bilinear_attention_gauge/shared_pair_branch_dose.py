"""Small-dose QK edit response fidelity of semantic and branch-aware cuts."""

import json
import numpy as np
import torch

from analyze_shared_pair_count import logits
from dual_joint_function_canonical import HERE
from shared_pair_branch_interventions import CONDITIONS,branch_tables
from train_shared_pair_count import SharedPairCountModel
from train_shared_pair_dense import examples_dense
from train_three_count_language import examples


DOSES=(.01,.05,.1,.25,.5,1.)


def relative_error(pred,true,mask=None):
    if mask is not None:pred=pred[mask];true=true[mask]
    return float(np.sum((pred-true)**2)/max(float(np.sum(true**2)),1e-20))


def run(n_test=10000):
    torch.set_num_threads(1)
    rows=[]
    for coding,source,prefix in (('paired',examples,'shared_pair_count'),
                                  ('dense',examples_dense,'shared_pair_dense')):
        for seed in range(10):
            model=SharedPairCountModel().double()
            model.load_state_dict(torch.load(HERE/f'{prefix}_seed{seed}.pt',
                                             map_location='cpu',weights_only=True))
            full=branch_tables(model,1.)
            _,_,v=np.linalg.svd(full['original'],full_matrices=False)
            ordinary=v[:3].T@v[:3]
            stack=np.vstack([full[k]/np.linalg.norm(full[k]) for k in CONDITIONS])
            _,_,w=np.linalg.svd(stack,full_matrices=False)
            branch_aware=w[:7].T@w[:7]
            projectors={'ordinary_rank3':ordinary,'all_four_rank7':branch_aware}
            bias=model.backbone.read.bias.detach().numpy()
            scale=model.logit_scale.detach().numpy()
            q=model.quadratic.detach().numpy()
            q=.5*(q+q.transpose(0,2,1))/scale[None,:,None]/scale[None,None,:]
            pb=model.parity_bias.detach().numpy()
            tokens,_,attrs=source(n_test,770000+seed,False)
            tokens=tokens.numpy();rare=attrs['heldout']
            original=logits(full['original'],bias,q,pb,tokens)
            original_projected={name:logits(full['original']@p,bias,q,pb,tokens)
                                for name,p in projectors.items()}
            for dose in DOSES:
                tables=branch_tables(model,dose)
                for condition in CONDITIONS[1:]:
                    true_table_delta=tables[condition]-full['original']
                    true=logits(tables[condition],bias,q,pb,tokens)-original
                    for name,p in projectors.items():
                        pred_table_delta=true_table_delta@p
                        reduced=logits(tables[condition]@p,bias,q,pb,tokens)
                        pred=reduced-original_projected[name]
                        rows.append(dict(coding=coding,seed=seed,dose=dose,
                            condition=condition,projector=name,
                            table_response_relative_mse=relative_error(
                                pred_table_delta,true_table_delta),
                            five_logit_response_relative_mse=relative_error(pred,true),
                            rare_five_logit_response_relative_mse=relative_error(
                                pred,true,rare)))
            print(coding,seed,flush=True)
    result=dict(doses=list(DOSES),n_test=n_test,
        edit='replace QK score s with (1-dose)*s + dose*1 before branch multiplication',
        evaluation='compare changes from each projector\'s own unedited baseline with physical-model change; no task labels',
        rows=rows)
    (HERE/'shared_pair_branch_dose.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()

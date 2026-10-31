"""Match physical-head reconstruction to learned sparse featurizers.

Choose a per-model top-k/top-(k+1) energy threshold solely on allowed-query
validation states. No XOR examples, labels, logits, or edit-path scores
enter calibration. Evaluate on identical held-out XOR test panels.
"""

import json

import numpy as np
import torch

from contextual_composition_causal_prior import evaluate, losses, make_panel
from contextual_composition_reduction import head_table
from contextual_composition_shared_groups import FlatSparse, hard_topk
from dual_joint_function_canonical import HERE
from train_contextual_composition_attention import ContextualCompositionAttention


TARGETS=('reconstruction_baseline','output_beta0.3','edit_gamma0.1')


class PhysicalEnergyTopKK1(torch.nn.Module):
    def __init__(self,k,threshold):
        super().__init__();self.k=int(k);self.threshold=float(threshold)

    def forward(self,h):
        values=h.abs()
        lower=hard_topk(values,self.k)
        upper=hard_topk(values,self.k+1)
        next_energy=values.topk(self.k+1,dim=1).values[:,-1].square()
        use_next=(next_energy>self.threshold).float()[:,None]
        mask=lower+use_next*(upper-lower)
        return h*mask,dict(mean_active=float(mask.sum(1).mean()))


def calibrate(val,target):
    h=torch.cat((val['h'],val['h_edit'])).numpy().astype(np.float64)
    sorted_energy=np.sort(h*h,axis=1)[:,::-1]
    total=float(np.sum(sorted_energy))
    errors=[float(np.sum(sorted_energy[:,k:]))/total for k in range(1,9)]
    k=next((j for j in range(1,8) if errors[j-1]>=target>=errors[j]),None)
    assert k is not None,(target,errors)
    lower_residual=errors[k-1]*total
    upper_residual=errors[k]*total
    gain=sorted_energy[:,k]
    needed=lower_residual-target*total
    if needed<=0:
        threshold=float('inf')
    elif needed>=gain.sum():
        threshold=-1.
    else:
        order=np.sort(gain)[::-1]
        n_selected=int(np.searchsorted(np.cumsum(order),needed,side='left'))+1
        threshold=(float((order[n_selected-1]+order[n_selected])/2)
                   if n_selected<len(order) else -1.)
    model=PhysicalEnergyTopKK1(k,threshold)
    with torch.no_grad():
        original,_=model(val['h'])
        edited,_=model(val['h_edit'])
    actual=float(((original-val['h']).square().sum()+
                  (edited-val['h_edit']).square().sum())/
                 (val['h'].square().sum()+val['h_edit'].square().sum()))
    return model,dict(target_allowed_validation_head_error=target,
                      achieved_allowed_validation_head_error=actual,
                      lower_physical_heads=k,
                      physical_lower_error=lower_residual/total,
                      physical_upper_error=upper_residual/total,
                      threshold=threshold,
                      mean_active_validation=float(np.mean(k+(gain>threshold))))


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(10):
        native=ContextualCompositionAttention()
        native.load_state_dict(torch.load(HERE/f'contextual_composition_attention_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        native.eval();native.requires_grad_(False)
        table=head_table(native)
        val=make_panel(native,table,8000,20263600+seed,fit_only=True)
        test=make_panel(native,table,20000,20263700+seed)
        shifted=make_panel(native,table,20000,20263800+seed,shifted=True)
        methods={}
        for name in TARGETS:
            fitted=FlatSparse(12)
            suffix=('flat_12' if name=='reconstruction_baseline' else name)
            source=('contextual_composition_groups_seed' if name=='reconstruction_baseline'
                    else 'contextual_composition_causal_seed')
            fitted.load_state_dict(torch.load(HERE/f'{source}{seed}_{suffix}.pt',
                                              map_location='cpu',weights_only=True))
            fitted.eval()
            with torch.no_grad():
                target=float(losses(fitted,native,val)[0])
            physical,calibration=calibrate(val,target)
            methods[name]=dict(calibration=calibration,
                learned=dict(test=evaluate(fitted,native,test),
                             shifted=evaluate(fitted,native,shifted)),
                matched_physical=dict(test=evaluate(physical,native,test),
                                      shifted=evaluate(physical,native,shifted)))
        rows.append(dict(seed=seed,panel='exploratory' if seed<5 else 'confirmation',
                         methods=methods))
        print(seed,{name:round(v['calibration']['mean_active_validation'],3)
                    for name,v in methods.items()},flush=True)
    summary={}
    for panel,seeds in (('exploratory',range(5)),('confirmation',range(5,10))):
        summary[panel]={}
        selected=[rows[i] for i in seeds]
        for name in TARGETS:
            summary[panel][name]={}
            for side in ('learned','matched_physical'):
                summary[panel][name][side]={}
                for split in ('test','shifted'):
                    summary[panel][name][side][split]={metric:float(np.mean([
                        row['methods'][name][side][split]['xor'][metric]
                        for row in selected]))
                        for metric in ('accuracy','head_relative_mse',
                                       'relative_logit_mse','edit_path_relative_mse',
                                       'edit_sign_agreement')}
            summary[panel][name]['mean_physical_active_validation']=float(np.mean([
                row['methods'][name]['calibration']['mean_active_validation']
                for row in selected]))
    result=dict(description='Physical top-k/top-(k+1) energy gate calibrated separately for each model and learned featurizer to match paired-head reconstruction error on primitive+AND validation. k and threshold use head energy only, no labels/logits/XOR. Independently evaluate calibrated physical and learned featurizer on identical uniform and shifted XOR panels, seeds 0–4 exploratory, seeds 5–9 independent confirmation. Achieved XOR head errors can differ after the whole-family shift.',
                target_names=TARGETS,rows=rows,summary=summary)
    (HERE/'contextual_composition_matched_physical.json').write_text(
        json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()

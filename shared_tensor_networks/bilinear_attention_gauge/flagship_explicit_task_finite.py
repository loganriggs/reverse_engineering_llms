"""Finite full-model QK1-key edits on frozen explicit-task axes and positions."""

import itertools
import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_explicit_task_gauge import BASIS_NAME, FAMILIES, STEM
from flagship_full_context_sensitivity_gauge import SensitivityEdit
from flagship_full_model_eval import GPT, GPTConfig, logits
from flagship_matrix_free import HD, SNAPSHOT
from flagship_transfer import ART


DOSE=.5


def run(stem=STEM):
    torch.set_num_threads(4)
    meta=json.loads((ART/f'{stem}.json').read_text())
    selection=(json.loads((ART/f'{stem}_analysis.json').read_text())
               ['axis_selection']['all_families_oracle'])
    arrays=np.load(ART/f'{stem}.npz')
    rotation=np.load(ART/f'{stem}_rotations.npz')['all_families_oracle']
    basis=np.load(ART/BASIS_NAME)['basis']
    assert basis.shape==(9*HD,256)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    editor=SensitivityEdit(model,meta['max_prompt_length'],basis)
    directions={family:torch.from_numpy((basis@rotation[:,selection[family]['axis']]).astype(np.float32))
                for family in FAMILIES}
    slopes=[];analytic=[];kls=[];margins=[];fallback_count=0
    test_rows=meta['rows']['test']
    for i,row in enumerate(test_rows):
        idx=torch.tensor(row['prompt_ids'],dtype=torch.long)[None,:]
        editor.set_shared(basis)
        with torch.no_grad():
            base=logits(model,idx)[0,-1].float()
        base_margin=float(base[row['correct_id']]-base[row['foil_id']])
        assert abs(base_margin-float(arrays['test_cut_margin'][i]))<2e-4
        base_prob=F.log_softmax(base,dim=-1)
        case_slopes=[];case_grad=[];case_kl=[]
        for family in FAMILIES:
            chosen=selection[family]
            raw_position=len(row['prompt_ids'])-1-chosen['offset_from_end']
            position=max(0,raw_position)
            fallback_count+=int(raw_position<0)
            direction=directions[family]
            result=[];kl=[]
            for sign in (1.,-1.):
                editor.injection=(position,direction,sign*DOSE)
                with torch.no_grad():
                    value=logits(model,idx)[0,-1].float()
                result.append(float(value[row['correct_id']]-value[row['foil_id']]))
                kl.append(float(torch.sum(base_prob.exp()*(base_prob-F.log_softmax(value,dim=-1)))))
            editor.injection=None
            case_slopes.append((result[0]-result[1])/(2*DOSE))
            case_kl.append(float(np.mean(kl)))
            projected=arrays['test_gradients'][i,position].astype(np.float64)
            case_grad.append(float(projected@rotation[:,chosen['axis']]))
        slopes.append(case_slopes);analytic.append(case_grad)
        kls.append(case_kl);margins.append(base_margin)
        if (i+1)%40==0:print('finite',i+1,len(test_rows),flush=True)
    slopes=np.asarray(slopes);analytic=np.asarray(analytic);kls=np.asarray(kls)
    labels=arrays['test_family']
    energy=slopes*slopes
    fraction=energy/np.maximum(energy.sum(1,keepdims=True),1e-30)
    per_family={}
    for i,family in enumerate(FAMILIES):
        own=labels==i;others=labels!=i
        per_family[family]={
            'n':int(own.sum()),
            'own_axis_mean_abs_slope':float(np.abs(slopes[own,i]).mean()),
            'other_families_mean_abs_slope':float(np.abs(slopes[others,i]).mean()),
            'own_vs_other_abs_slope_ratio':float(np.abs(slopes[own,i]).mean()/
                max(np.abs(slopes[others,i]).mean(),1e-30)),
            'own_axis_mean_selected_energy_fraction':float(fraction[own,i].mean()),
            'own_axis_mean_kl':float(kls[own,i].mean()),
            'mean_native_correct_margin':float(np.mean(np.asarray(margins)[own])),
        }
    observed=float(np.mean([fraction[labels==i,i].mean() for i in range(len(FAMILIES))]))
    permuted=[]
    for mapping in itertools.permutations(range(len(FAMILIES))):
        permuted.append(float(np.mean([fraction[labels==i,mapping[i]].mean()
                                       for i in range(len(FAMILIES))])))
    result={'stem':stem,'dose':DOSE,'families':FAMILIES,
            'axis_selection':selection,
            'n_test_prompts':len(test_rows),
            'short_prompt_position_fallbacks':fallback_count,
            'max_finite_vs_analytic_slope_abs_error':float(np.max(np.abs(slopes-analytic))),
            'mean_finite_vs_analytic_slope_abs_error':float(np.mean(np.abs(slopes-analytic))),
            'per_family':per_family,
            'mean_own_axis_energy_fraction':observed,
            'permutation_null_mean':float(np.mean(permuted)),
            'permutation_p_ge_observed':float(np.mean(np.asarray(permuted)>=observed)),
            'caveat':'off-manifold QK1-key perturbation in a fixed reduced model; axes saw all task families on train; no claim of independent physical heads'}
    (ART/f'{stem}_finite.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ART/f'{stem}_finite.npz',slopes=slopes,
                        analytic_slopes=analytic,kls=kls,
                        family=labels,margins=np.asarray(margins))
    print('own fraction',observed,'permutation p',result['permutation_p_ge_observed'],flush=True)
    return result


if __name__=='__main__':
    run()

"""Finite selected-axis patch validation for exact rank-4 head gauges."""

import json

import numpy as np
import torch

from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_rank_pilot import HeadCut, final_logits
from flagship_full_model_eval import GPT, GPTConfig
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

VARIANTS=('svd','weight__all','edit__all','effect__all',
          'edit__target_only','effect__target_only',
          'edit__exclude_target','effect__exclude_target')


def margin(model,ids,a,b):
    value=final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:])[0]
    return float(value[a]-value[b])


def run():
    torch.set_num_threads(4)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probes={layer:HeadCut(model.transformer.h[layer].attn)
            for layer in sorted(set(layer for layer,_ in TARGETS.values()))}
    capture=np.load(ART/'later_head_gauge_capture.npz')
    metadata=json.loads((ART/'later_head_gauge_capture.json').read_text())
    bases=np.load(ART/'later_head_rank_pilot_bases.npz')
    rotations=np.load(ART/'later_head_exact_gauge_rotations.npz')
    records=[];summary={}
    with torch.no_grad():
        for target,(layer,head) in TARGETS.items():
            p=probes[layer];p.head=head
            b=torch.tensor(bases[f'{target}__weight'][:,:4].copy(),dtype=torch.float32)
            p.basis=b
            train_d=capture[f'{target}__train__d']
            train_g=capture[f'{target}__train__g']
            train_rows=metadata['rows'][f'{target}__train']
            target_train=np.array([x['family']==target for x in train_rows])
            hold_d=capture[f'{target}__holdout__d']
            hold_raw=capture[f'{target}__holdout__raw_delta']
            hold_rows=metadata['rows'][f'{target}__holdout']
            part=[i for i,x in enumerate(hold_rows) if x['family']==target]
            controls=[f'random_{i}' for i in range(64)]
            variants=VARIANTS+tuple(controls)
            chosen={}
            for variant in variants:
                r=np.eye(4) if variant=='svd' else rotations[f'{target}__{variant}']
                selection_mask=(~target_train if variant.endswith('__exclude_target')
                                else target_train)
                effects=(train_d[selection_mask]@r)*(train_g[selection_mask]@r)
                chosen[variant]=int(np.argmax(np.mean(np.abs(effects),axis=0)))
                assert np.max(np.abs(r.T@r-np.eye(4)))<1e-6
            for i in part:
                item=hold_rows[i]
                base=margin(model,item['before_ids'],item['answer_a'],item['answer_b'])
                delta=torch.tensor(hold_raw[i],dtype=torch.float32)[None,:]
                projected=(delta@b)@b.T
                p.injection=projected
                full=margin(model,item['before_ids'],item['answer_a'],item['answer_b'])-base
                p.injection=None
                for variant in variants:
                    r=np.eye(4) if variant=='svd' else rotations[f'{target}__{variant}']
                    axis=chosen[variant]
                    u=b@torch.tensor(r[:,axis].copy(),dtype=torch.float32)
                    coefficient=float(delta@u)
                    p.injection=coefficient*u[None,:]
                    effect=margin(model,item['before_ids'],item['answer_a'],item['answer_b'])-base
                    p.injection=None
                    records.append(dict(target=target,variant=variant,axis=axis,
                        holdout_index=i,full_rank4_path_effect=full,
                        single_axis_effect=effect,
                        absolute_error=abs(effect-full),
                        sign_match=bool(np.sign(effect)==np.sign(full))))
                print(target,i,'full',round(full,3),flush=True)
            for variant in variants:
                subset=[x for x in records if x['target']==target and x['variant']==variant]
                full=np.array([x['full_rank4_path_effect'] for x in subset])
                one=np.array([x['single_axis_effect'] for x in subset])
                summary[f'{target}__{variant}']=dict(n=len(subset),selected_axis=chosen[variant],
                    mean_abs_full_path=float(np.mean(np.abs(full))),
                    mean_abs_single_axis=float(np.mean(np.abs(one))),
                    relative_mae=float(np.mean(np.abs(one-full))/np.mean(np.abs(full))),
                    sign_matches=int(np.sum(np.sign(one)==np.sign(full))))
            p.basis=None
    result=dict(description='One selected rotated coordinate injected at final position; supervised target-axis selection for ordinary and random variants, non-target selection for excluded-target variants',
                selection_caveat='Supervised axis selection, including the random controls, is not unsupervised discovery. Exclude-target variants fit rotation and select axis without target-family examples.',
                summary=summary,rows=records)
    (ART/'later_head_exact_gauge_finite.json').write_text(json.dumps(result,indent=2)+'\n')
    for target in TARGETS:
        print(target,[(v,round(summary[f'{target}__{v}']['relative_mae'],3)) for v in VARIANTS],flush=True)


if __name__=='__main__':run()

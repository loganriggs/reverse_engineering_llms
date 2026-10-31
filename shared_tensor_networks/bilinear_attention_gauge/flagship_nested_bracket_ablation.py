"""Remove frozen bracket-group directions on standard and nested prompts."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_bracket_group_competent_transfer import margin
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_rank_pilot import HeadCut,final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


def head_state(model,probe,ids):
    final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:],stop_at=13)
    start=8*probe.head_dim
    return probe.last_input[0,-1,start:start+probe.head_dim].clone()


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    a=tokenizer.encode(')',add_special_tokens=False).ids[0]
    b=tokenizer.encode(']',add_special_tokens=False).ids[0]
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probe=HeadCut(model.transformer.h[13].attn);probe.head=8
    bases=np.load(ART/'head_joint_moment_fit_bases.npz')
    joint=torch.tensor(bases['bracket_closure__lambda_0.1'].copy(),dtype=torch.float32)
    orientations=np.load(ART/'joint_bracket_legibility_rotations.npz')
    choice=json.loads((ART/'joint_bracket_legibility.json').read_text())
    selected=choice['selectivity']['raw_pca']['selected_axes']
    remaining=[i for i in range(4) if i not in selected]
    r=torch.tensor(orientations['raw_pca'].copy(),dtype=torch.float32)
    directions={'joint_rank2':joint@r[:,selected],
                'joint_other2':joint@r[:,remaining],
                'joint_rank4':joint,
                'full_head':torch.eye(128)}
    for i in range(choice['random_controls']):
        q=torch.tensor(orientations[f'random_{i:02d}'].copy(),dtype=torch.float32)
        selected_i=choice['selectivity'][f'random_{i:02d}']['selected_axes']
        directions[f'random_{i:02d}']=joint@q[:,selected_i]
    assert all(torch.linalg.norm(x.T@x-torch.eye(x.shape[1]))<1e-5
               for x in directions.values())
    nested=json.loads((ART/'nested_bracket_transfer.json').read_text())
    standard=json.loads((ART/'head_joint_bracket_confirmation.json').read_text())
    cases=[dict(panel=f'nested_{x["kind"]}',base_ids=x['base_ids'],
                edited_ids=x['edited_ids']) for x in nested['rows']]
    cases += [dict(panel='standard',base_ids=x['base_ids'],
                   edited_ids=x['edited_ids']) for x in standard['rows']]
    assert len(cases)==48
    rows=[]
    with torch.no_grad():
        for case in cases:
            margins={}
            for prompt_name,ids in (('base',case['base_ids']),
                                    ('edited',case['edited_ids'])):
                state=head_state(model,probe,ids)[None,:]
                native=margin(model,ids,a,b)
                intervention={}
                probe.basis=torch.eye(probe.head_dim)
                for name,direction in directions.items():
                    probe.injection=-(state@direction)@direction.T
                    intervention[name]=margin(model,ids,a,b)
                    probe.injection=None
                probe.basis=None
                margins[prompt_name]=dict(native=native,ablated=intervention)
            expected=(1,-1) if case['panel']!='nested_inner' else (-1,1)
            native_correct=(margins['base']['native']*expected[0]>0 and
                            margins['edited']['native']*expected[1]>0)
            ablated_correct={name:(margins['base']['ablated'][name]*expected[0]>0 and
                                   margins['edited']['ablated'][name]*expected[1]>0)
                             for name in directions}
            rows.append(dict(panel=case['panel'],base_ids=case['base_ids'],
                edited_ids=case['edited_ids'],native_correct=native_correct,
                ablated_correct=ablated_correct,margins=margins))
            if len(rows)%12==0:print('ablation',len(rows),'/',len(cases),flush=True)
    summary=[]
    for panel in ('standard','nested_inner','nested_outer','all'):
        part=rows if panel=='all' else [x for x in rows if x['panel']==panel]
        for name in directions:
            native=np.array([m['native'] for x in part for m in x['margins'].values()])
            changed=np.array([m['ablated'][name] for x in part for m in x['margins'].values()])
            summary.append(dict(panel=panel,method=name,pairs=len(part),
                native_pairs_correct=sum(x['native_correct'] for x in part),
                ablated_pairs_correct=sum(x['ablated_correct'][name] for x in part),
                native_prompt_mean_abs_margin=float(np.mean(np.abs(native))),
                margin_shift_mean_abs=float(np.mean(np.abs(changed-native))),
                margin_shift_over_native_abs_mean=float(np.mean(np.abs(changed-native))/
                    max(np.mean(np.abs(native)),1e-20))))
    result=dict(description='Native-model final-position L13 H8 head-state ablation: remove frozen joint two-axis group, complementary two axes, full joint rank four, full head, and matched random rank-two groups on 24 standard and 24 nested bracket pairs',
        intervention='subtract projection of native final-position head state onto selected subspace before c_proj',
        rows=rows,summary=summary)
    (ART/'nested_bracket_ablation.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in summary:
        if row['panel']!='all' and not row['method'].startswith('random_'):
            print(row,flush=True)


if __name__=='__main__':run()

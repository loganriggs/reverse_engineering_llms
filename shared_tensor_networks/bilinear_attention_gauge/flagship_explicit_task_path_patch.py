"""Minimal token-swap red team for explicit-task first-layer QK1-key axes."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_gauge import BASIS_NAME, STEM
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_context_sensitivity_gauge import SensitivityEdit
from flagship_full_model_eval import GPT, GPTConfig, logits
from flagship_matrix_free import HD, SNAPSHOT
from flagship_transfer import ART


class AttentionOutputPatch:
    def __init__(self,attention):
        self.native=attention.forward
        self.delta=None
        self.last_output=None
        attention.forward=self.forward

    def forward(self,x,v1=None):
        output,carried=self.native(x,v1)
        self.last_output=output.detach().clone()
        if self.delta is not None:
            output=output+self.delta
        return output,carried


def proposed_pairs():
    out=[]
    capitals=[('France',' Paris','Germany',' Berlin'),
              ('Italy',' Rome','Japan',' Tokyo'),
              ('Spain',' Madrid','China',' Beijing'),
              ('Canada',' Ottawa','Egypt',' Cairo'),
              ('Russia',' Moscow','Greece',' Athens'),
              ('India',' Delhi','Portugal',' Lisbon')]
    for a,answer_a,b,answer_b in capitals:
        out.append(('country_capital',f'The capital of {a} is',
                    f'The capital of {b} is',answer_a,answer_b))
    nouns=[('key','keys','cabinets'),('dog','dogs','cats'),
           ('book','books','shelves'),('car','cars','garages'),
           ('student','students','teachers')]
    for singular,plural,distractor in nouns:
        out.append(('subject_verb_agreement',
                    f'The {singular} near the {distractor}',
                    f'The {plural} near the {distractor}',' is',' are'))
    for a,b in ((1,2),(3,4),(5,6),(7,8),(2,9),(9,1),(4,7),(6,3)):
        out.append(('bracket_closure',f'({a} + {b}',f'[{a} + {b}',')',']'))
    for a,opposite,b,reverse in [('hot',' cold','cold',' hot'),
                                 ('big',' small','small',' big'),
                                 ('fast',' slow','slow',' fast'),
                                 ('dark',' light','light',' dark')]:
        out.append(('antonym',f'The opposite of {a} is',
                    f'The opposite of {b} is',opposite,reverse))
    return out


def score(model,tokens,candidate_a,candidate_b):
    value=logits(model,torch.tensor(tokens,dtype=torch.long)[None,:])[0,-1].float()
    return float((value[candidate_a]-value[candidate_b]).detach())


def run(stem=STEM):
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    meta=json.loads((ART/f'{stem}.json').read_text())
    selected=json.loads((ART/f'{stem}_analysis.json').read_text())
    selected=selected['axis_selection']['all_families_oracle']
    rotation=np.load(ART/f'{stem}_rotations.npz')['all_families_oracle']
    basis=np.load(ART/BASIS_NAME)['basis']
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    editor=SensitivityEdit(model,meta['max_prompt_length'],basis)
    attention_patch=AttentionOutputPatch(model.transformer.h[0].attn)
    rows=[]
    for family,base_prompt,edited_prompt,base_answer,edited_answer in proposed_pairs():
        before=tokenizer.encode(base_prompt,add_special_tokens=False).ids
        after=tokenizer.encode(edited_prompt,add_special_tokens=False).ids
        answer_a=tokenizer.encode(base_answer,add_special_tokens=False).ids
        answer_b=tokenizer.encode(edited_answer,add_special_tokens=False).ids
        assert len(answer_a)==len(answer_b)==1
        assert len(before)==len(after)<=meta['max_prompt_length']
        changed=[i for i,(a,b) in enumerate(zip(before,after)) if a!=b]
        assert len(changed)==1,(family,base_prompt,edited_prompt,before,after)
        position=changed[0]
        editor.reset()
        native_base=score(model,before,answer_a[0],answer_b[0])
        native_edit=score(model,after,answer_a[0],answer_b[0])
        editor.set_shared(basis)
        cut_base=score(model,before,answer_a[0],answer_b[0])
        pre_base=editor.last_pre.detach()[0,position].clone()
        attention_base=attention_patch.last_output.clone()
        cut_edit=score(model,after,answer_a[0],answer_b[0])
        pre_edit=editor.last_pre.detach()[0,position].clone()
        attention_edit=attention_patch.last_output.clone()
        attention_patch.delta=attention_edit-attention_base
        full_attention_path=score(model,before,answer_a[0],answer_b[0])
        attention_patch.delta=None
        delta=pre_edit-pre_base
        editor.injection=(position,delta,1.)
        full_path=score(model,before,answer_a[0],answer_b[0])
        editor.injection=None
        axis=int(selected[family]['axis'])
        axis_direction=torch.from_numpy((basis@rotation[:,axis]).astype(np.float32))
        coefficient=float((delta.reshape(-1)@axis_direction).detach())
        editor.injection=(position,coefficient*axis_direction.reshape(9,HD),1.)
        axis_path=score(model,before,answer_a[0],answer_b[0])
        editor.injection=None
        projected_delta=(delta.reshape(-1).numpy()@basis)
        total_projected_energy=float(np.sum(projected_delta**2))
        row={'family':family,'base_prompt':base_prompt,'edited_prompt':edited_prompt,
             'changed_position':position,'changed_token_ids':[before[position],after[position]],
             'selected_axis':axis,
             'native_base_margin':native_base,'native_edited_margin':native_edit,
             'cut_base_margin':cut_base,'cut_edited_margin':cut_edit,
             'native_input_edit_margin_change':native_edit-native_base,
             'cut_input_edit_margin_change':cut_edit-cut_base,
             'full_layer0_attention_output_path_margin_change':full_attention_path-cut_base,
             'full_qk1_key_path_margin_change':full_path-cut_base,
             'selected_axis_path_margin_change':axis_path-cut_base,
             'selected_axis_fraction_of_projected_key_delta_energy':coefficient**2/
                 max(total_projected_energy,1e-30),
             'base_correct':native_base>0,'edited_correct':native_edit<0}
        rows.append(row)
    summary={}
    for family in sorted(set(row['family'] for row in rows)):
        part=[row for row in rows if row['family']==family]
        summary[family]={'n':len(part),
            'both_native_choices_correct':int(sum(row['base_correct'] and row['edited_correct'] for row in part)),
            'mean_abs_native_input_edit_change':float(np.mean([abs(row['native_input_edit_margin_change']) for row in part])),
            'mean_abs_full_qk1_key_path_change':float(np.mean([abs(row['full_qk1_key_path_margin_change']) for row in part])),
            'mean_abs_full_layer0_attention_output_path_change':float(np.mean([
                abs(row['full_layer0_attention_output_path_margin_change']) for row in part])),
            'mean_abs_selected_axis_path_change':float(np.mean([abs(row['selected_axis_path_margin_change']) for row in part])),
            'mean_selected_axis_projected_key_energy_fraction':float(np.mean([
                row['selected_axis_fraction_of_projected_key_delta_energy'] for row in part]))}
    result={'stem':stem,'description':'single-token semantic/syntactic swaps; patch only first-layer projected QK1 key at changed token',
            'summary':summary,'rows':rows,
            'caveat':'path-specific interventions in fixed rank256 reduced model; margin changes are not additive causal mediation fractions'}
    (ART/f'{stem}_path_patch.json').write_text(json.dumps(result,indent=2)+'\n')
    print(summary,flush=True)
    return result


if __name__=='__main__':
    run()

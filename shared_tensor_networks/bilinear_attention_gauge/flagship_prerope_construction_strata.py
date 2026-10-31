"""Same-token construction stress test for the first-layer lexical group."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import SNAPSHOT
from flagship_prerope_conditional_group_ablation import KeyConditionalEdit,kl_positions
from flagship_prerope_group_ablation import candidates
from flagship_transfer import ART

SEED=20261106
PER_STRATUM=8
STRATA=('as_comparison_first','as_comparison_second','as_role','as_clause',
        'as_well_first','as_well_second','so_degree','so_result','so_purpose')
GROUPS=('identity_0','haar_rank10_0')
DOSES=(.25,1.)


def draw(kind,rng):
    word=lambda values:str(rng.choice(values))
    if kind.startswith('as_comparison'):
        noun=word(('car','runner','bird','train','horse','plane'))
        other=word(('train','runner','bus','hawk','horse','boat'))
        adjective=word(('fast','bright','quiet','strong','small','loud'))
        ending=word(('during the contest','when the race ended',
                     'in the story','on that day'))
        return f'The {noun} was as {adjective} as the {other} {ending}.'
    if kind=='as_role':
        name=word(('Mia','Noah','Ava','Liam','Emma','Lucas'))
        job=word(('doctor','teacher','guide','driver','writer','nurse'))
        place=word(('city','village','school','hospital','station'))
        return f'{name} worked as a {job} in the {place} for years.'
    if kind=='as_clause':
        event=word(('rain stopped','sun rose','clock rang','door opened',
                    'train arrived','lights faded'))
        action=word(('children returned home','workers left the room',
                     'team began the task','visitors waited outside'))
        return f'Later, as the {event}, the {action}.'
    if kind.startswith('as_well'):
        subject=word(('team','family','group','teacher','visitor'))
        first=word(('apples','bread','books','water','chairs'))
        second=word(('bread','flowers','notes','fruit','maps'))
        place=word(('picnic','meeting','classroom','station'))
        return f'The {subject} brought {first} as well as {second} to the {place}.'
    if kind=='so_degree':
        noun=word(('music','storm','bell','engine','crowd','noise'))
        adjective=word(('loud','strong','bright','quiet','cold','strange'))
        result=word(('everyone left the room','the children ran outside',
                     'people stopped talking','we could not hear'))
        return f'The {noun} was so {adjective} that {result}.'
    if kind=='so_result':
        event=word(('rained','snowed','was late','became dark','grew cold'))
        result=word(('road became wet','team went home','driver slowed down',
                     'children stayed inside','lights came on'))
        return f'It {event}, so the {result}.'
    if kind=='so_purpose':
        action=word(('left early','spoke softly','opened the door',
                     'wrote a note','waited outside'))
        purpose=word(('train would not wait','child could sleep',
                      'teacher would understand','room would stay quiet'))
        return f'They {action} so that the {purpose}.'
    raise ValueError(kind)


def make_cases(tokenizer):
    rng=np.random.default_rng(SEED)
    result=[];seen=set()
    for stratum in STRATA:
        attempts=0;made=0
        token=355 if stratum.startswith('as_') else 523
        ordinal=1 if stratum.endswith('second') else 0
        expected=2 if stratum.startswith(('as_comparison','as_well')) else 1
        while made<PER_STRATUM:
            attempts+=1;assert attempts<10000,(stratum,made)
            text=draw(stratum,rng)
            ids=tokenizer.encode(text,add_special_tokens=False).ids
            positions=[i for i,x in enumerate(ids) if x==token]
            if len(positions)!=expected or text in seen:continue
            seen.add(text)
            result.append(dict(stratum=stratum,target_token_id=token,
                occurrence=ordinal,target_position=positions[ordinal],
                text=text,ids=ids))
            made+=1
        print('generated',stratum,made,'attempts',attempts,flush=True)
    return result


def run():
    torch.set_num_threads(4)
    cases=make_cases(Tokenizer.from_file(TOKENIZER))
    base,effects,details=candidates()
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    editor=KeyConditionalEdit(model)
    rows=[];identity_gate=None
    with torch.no_grad():
        for length in sorted({len(x['ids']) for x in cases}):
            editor.set_matrices(base,effects[GROUPS[0]],length)
            transported_base=editor.base_matrices
            transported={GROUPS[0]:editor.effect_matrices}
            editor.set_matrices(base,effects[GROUPS[1]],length)
            transported[GROUPS[1]]=editor.effect_matrices
            editor.base_matrices=transported_base
            indices=[i for i,x in enumerate(cases) if len(x['ids'])==length]
            for start in range(0,len(indices),4):
                group=indices[start:start+4]
                ids=torch.tensor([cases[i]['ids'] for i in group],dtype=torch.long)
                mask=torch.zeros_like(ids,dtype=torch.bool)
                for j,i in enumerate(group):mask[j,cases[i]['target_position']]=True
                editor.key_mask=torch.zeros_like(mask)
                editor.fraction=1.
                reference=logits(model,ids)
                if identity_gate is None:
                    repeat=logits(model,ids)
                    identity_gate=float((repeat-reference).abs().max())
                    assert identity_gate<1e-6
                for name in GROUPS:
                    editor.effect_matrices=transported[name]
                    editor.key_mask=mask
                    for dose in DOSES:
                        editor.fraction=dose
                        candidate=logits(model,ids)
                        kl=kl_positions(reference,candidate)
                        for j,i in enumerate(group):
                            position=cases[i]['target_position']
                            after=kl[j,position:]
                            rows.append(dict(index=i,stratum=cases[i]['stratum'],
                                target_token_id=cases[i]['target_token_id'],
                                target_position=position,group=name,dose=dose,
                                kl_after_key_mean=float(np.mean(after)),
                                kl_at_key=float(kl[j,position]),
                                total_kl_sum=float(np.sum(kl[j])),
                                removed_head_energy=float(editor.removed_head_energy[j])))
                print('length',length,'processed',len(set(r['index'] for r in rows)),
                      '/',len(cases),flush=True)
    result=dict(description='Curated same-token construction stress panel for a fixed first-layer pre-RoPE QK1 response group. Nine strata, eight prompt templates each, selected occurrence of token ID 355 (as) or 523 (so). Compare identity-start response group and one rank-ten Haar control at 25% and full deletion. The reduced cut and groups predate these prompts. No answer-level task competence is asserted; construction labels are designed for validation. KL is measured after the selected key and removed pre-projection head energy recorded for dose analysis.',
        seed=SEED,strata=STRATA,per_stratum=PER_STRATUM,
        groups=GROUPS,doses=DOSES,identity_gate_max_abs_logit=identity_gate,
        group_details={name:details[name] for name in GROUPS},cases=cases,rows=rows)
    (ART/'prerope_construction_strata.json').write_text(json.dumps(result,indent=2)+'\n')
    for stratum in STRATA:
        for name in GROUPS:
            part=[r for r in rows if r['stratum']==stratum and r['group']==name
                  and r['dose']==1.]
            print(stratum,name,'afterKL',float(np.mean([r['kl_after_key_mean']
                for r in part])),'energy',float(np.mean([r['removed_head_energy']
                for r in part])),flush=True)


if __name__=='__main__':run()

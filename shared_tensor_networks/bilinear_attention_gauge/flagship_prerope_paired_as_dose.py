"""Paired first/second 'as' edits with matched removed head energy."""

import json
import os

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import SNAPSHOT
from flagship_prerope_conditional_group_ablation import KeyConditionalEdit,kl_positions
from flagship_prerope_construction_strata import draw
from flagship_prerope_group_ablation import candidates
from flagship_transfer import ART

SEED=20261107
N_PER_KIND=12
ENERGY_FRACTION=.25
KINDS=('comparison','as_well')
GROUP_NAME=os.environ.get('ATTN_GROUP','identity_0')


def make_cases(tokenizer):
    rng=np.random.default_rng(SEED)
    old=json.loads((ART/'prerope_construction_strata.json').read_text())
    seen={r['text'] for r in old['cases']}
    cases=[]
    for kind in KINDS:
        made=0;attempts=0
        source='as_comparison_first' if kind=='comparison' else 'as_well_first'
        while made<N_PER_KIND:
            attempts+=1;assert attempts<10000,(kind,made)
            text=draw(source,rng)
            ids=tokenizer.encode(text,add_special_tokens=False).ids
            positions=[i for i,x in enumerate(ids) if x==355]
            if len(positions)!=2 or text in seen:continue
            seen.add(text)
            cases.append(dict(kind=kind,text=text,ids=ids,
                              first_position=positions[0],second_position=positions[1]))
            made+=1
        print('generated',kind,made,'attempts',attempts,flush=True)
    return cases


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
    rows=[]
    with torch.no_grad():
        for length in sorted({len(r['ids']) for r in cases}):
            editor.set_matrices(base,effects[GROUP_NAME],length)
            indices=[i for i,r in enumerate(cases) if len(r['ids'])==length]
            for start in range(0,len(indices),4):
                group=indices[start:start+4]
                ids=torch.tensor([cases[i]['ids'] for i in group],dtype=torch.long)
                editor.key_mask=torch.zeros_like(ids,dtype=torch.bool)
                editor.fraction=1.
                reference=logits(model,ids)
                positions=np.array([[cases[i]['first_position'],
                                     cases[i]['second_position']] for i in group])
                full=[]
                for occurrence in (0,1):
                    mask=torch.zeros_like(ids,dtype=torch.bool)
                    for j in range(len(group)):mask[j,int(positions[j,occurrence])]=True
                    editor.key_mask=mask
                    candidate=logits(model,ids)
                    kl=kl_positions(reference,candidate)
                    full.append(dict(energy=editor.removed_head_energy.copy(),
                                     output_energy=editor.removed_head_output_energy.copy(),
                                     kl=kl))
                energy=np.stack([x['energy'] for x in full],axis=1)
                target=ENERGY_FRACTION*np.min(energy,axis=1)
                assert np.all(target>1e-10)
                matched=[]
                for occurrence in (0,1):
                    mask=torch.zeros_like(ids,dtype=torch.bool)
                    for j in range(len(group)):mask[j,int(positions[j,occurrence])]=True
                    editor.key_mask=mask
                    fraction=np.sqrt(target/energy[:,occurrence])
                    assert np.all(fraction<=.5+1e-6)
                    editor.fraction=torch.tensor(fraction,dtype=torch.float32)[:,None,None]
                    candidate=logits(model,ids)
                    matched.append(dict(fraction=fraction,
                        energy=editor.removed_head_energy.copy(),
                        output_energy=editor.removed_head_output_energy.copy(),
                        kl=kl_positions(reference,candidate),
                        logit_delta=(candidate-reference).double()))
                for j,i in enumerate(group):
                    second=int(positions[j,1])
                    a=matched[0]['logit_delta'][j,second:].reshape(-1)
                    b=matched[1]['logit_delta'][j,second:].reshape(-1)
                    cosine=float((a@b)/(a.norm()*b.norm()).clamp_min(1e-30))
                    rows.append(dict(index=i,kind=cases[i]['kind'],
                        first_position=int(positions[j,0]),second_position=second,
                        full_first_energy=float(energy[j,0]),
                        full_second_energy=float(energy[j,1]),
                        full_first_output_energy=float(full[0]['output_energy'][j]),
                        full_second_output_energy=float(full[1]['output_energy'][j]),
                        target_energy=float(target[j]),
                        full_first_common_suffix_kl=float(np.mean(full[0]['kl'][j,second:])),
                        full_second_common_suffix_kl=float(np.mean(full[1]['kl'][j,second:])),
                        matched_first_fraction=float(matched[0]['fraction'][j]),
                        matched_second_fraction=float(matched[1]['fraction'][j]),
                        matched_first_energy=float(matched[0]['energy'][j]),
                        matched_second_energy=float(matched[1]['energy'][j]),
                        matched_first_output_energy=float(matched[0]['output_energy'][j]),
                        matched_second_output_energy=float(matched[1]['output_energy'][j]),
                        matched_first_common_suffix_kl=float(np.mean(matched[0]['kl'][j,second:])),
                        matched_second_common_suffix_kl=float(np.mean(matched[1]['kl'][j,second:])),
                        matched_common_suffix_logit_cosine=cosine))
                editor.fraction=1.
                print('length',length,'processed',len(rows),'/',len(cases),flush=True)
    result=dict(description='Fresh paired sentences with exactly two occurrences of token ID355 as. For each sentence, delete the selected fixed pre-RoPE QK1 group at first or second occurrence. Match removed pre-projection head energy at 25% of the smaller full-dose energy, so both fractions <= .5. Compare final-logit KL only on the common suffix from the second as onward. The cut and groups precede these prompts; no model/task training. Designed templates distinguish comparison versus as-well-as, not a natural-language population sample.',
        seed=SEED,kinds=KINDS,n_per_kind=N_PER_KIND,
        target_energy_fraction_of_min_full=ENERGY_FRACTION,
        group_name=GROUP_NAME,group_details=details[GROUP_NAME],cases=cases,rows=rows)
    (ART/f'prerope_paired_as_dose_{GROUP_NAME}.json').write_text(
        json.dumps(result,indent=2)+'\n')
    for kind in KINDS:
        part=[r for r in rows if r['kind']==kind]
        print(kind,'full energy ratio',float(np.mean([r['full_first_energy']/r['full_second_energy'] for r in part])),
            'matched KL first',float(np.mean([r['matched_first_common_suffix_kl'] for r in part])),
            'second',float(np.mean([r['matched_second_common_suffix_kl'] for r in part])),
            'first>second',sum(r['matched_first_common_suffix_kl']>
                               r['matched_second_common_suffix_kl'] for r in part),flush=True)


if __name__=='__main__':run()

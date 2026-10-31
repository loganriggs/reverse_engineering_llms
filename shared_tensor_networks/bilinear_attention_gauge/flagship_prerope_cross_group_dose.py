"""Cross-group and cross-occurrence matched-energy lexical intervention."""

import json

import numpy as np
import torch

from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import SNAPSHOT
from flagship_prerope_conditional_group_ablation import KeyConditionalEdit,kl_positions
from flagship_prerope_group_ablation import candidates
from flagship_transfer import ART

GROUPS=('identity_0','haar_rank10_0','haar_rank10_1','haar_rank10_2')


def run():
    torch.set_num_threads(4)
    source={name:json.loads((ART/f'prerope_paired_as_dose_{name}.json').read_text())
            for name in GROUPS}
    cases=source[GROUPS[0]]['cases']
    assert all(source[name]['cases']==cases for name in GROUPS)
    assert len(cases)==24
    indexed={name:{r['index']:r for r in source[name]['rows']} for name in GROUPS}
    assert all(len(indexed[name])==len(cases) for name in GROUPS)
    full_energy=np.array([[
        [indexed[name][i]['full_first_output_energy'],
         indexed[name][i]['full_second_output_energy']]
        for name in GROUPS] for i in range(len(cases))])
    target=np.min(full_energy,axis=(1,2))
    assert np.all(target>0)
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
            transported={}
            for name in GROUPS:
                editor.set_matrices(base,effects[name],length)
                transported[name]=editor.effect_matrices
            editor.set_matrices(base,effects[GROUPS[0]],length)
            indices=[i for i,r in enumerate(cases) if len(r['ids'])==length]
            for start in range(0,len(indices),4):
                group=indices[start:start+4]
                ids=torch.tensor([cases[i]['ids'] for i in group],dtype=torch.long)
                editor.key_mask=torch.zeros_like(ids,dtype=torch.bool)
                editor.fraction=1.
                reference=logits(model,ids)
                for name_index,name in enumerate(GROUPS):
                    editor.effect_matrices=transported[name]
                    for occurrence in (0,1):
                        mask=torch.zeros_like(ids,dtype=torch.bool)
                        for j,i in enumerate(group):
                            position=(cases[i]['first_position'] if occurrence==0
                                      else cases[i]['second_position'])
                            mask[j,position]=True
                        editor.key_mask=mask
                        fraction=np.sqrt(target[group]/full_energy[group,name_index,occurrence])
                        assert np.all(fraction<=1+1e-6)
                        editor.fraction=torch.tensor(fraction,dtype=torch.float32)[:,None,None]
                        edited=logits(model,ids)
                        kl=kl_positions(reference,edited)
                        for j,i in enumerate(group):
                            common=cases[i]['second_position']
                            logit_delta=(edited[j,common:]-reference[j,common:]).double()
                            rows.append(dict(index=i,kind=cases[i]['kind'],
                                group=name,occurrence='first' if occurrence==0 else 'second',
                                target_output_energy=float(target[i]),fraction=float(fraction[j]),
                                removed_head_energy=float(editor.removed_head_energy[j]),
                                removed_head_output_energy=float(
                                    editor.removed_head_output_energy[j]),
                                common_suffix_kl=float(np.mean(kl[j,common:])),
                                common_suffix_logit_l2=float(logit_delta.norm())))
                editor.fraction=1.
                print('length',length,'processed',len(set(r['index'] for r in rows)),
                      '/',len(cases),flush=True)
    result=dict(description='On the same 24 fresh two-as sentences, match removed post-c_proj residual energy separately for every prompt across both as occurrences and four fixed QK1 groups. Target equals the minimum full-dose output energy among eight interventions on that prompt; all doses <=1. Evaluate final-logit KL on identical common suffix from second as onward. Identity response group versus three predeclared Haar rank-ten controls. Small target-energy cases may approach numerical floor; report high-energy subset separately.',
        groups=GROUPS,cases=cases,rows=rows,target_output_energy=target.tolist())
    (ART/'prerope_cross_group_dose_postoutput.json').write_text(json.dumps(result,indent=2)+'\n')
    for kind in ('comparison','as_well'):
        for name in GROUPS:
            a=[r for r in rows if r['kind']==kind and r['group']==name
               and r['occurrence']=='first' and r['target_output_energy']>=.5]
            b=[r for r in rows if r['kind']==kind and r['group']==name
               and r['occurrence']=='second' and r['target_output_energy']>=.5]
            print(kind,name,'n',len(a),'firstKL',float(np.mean([x['common_suffix_kl'] for x in a])),
                  'secondKL',float(np.mean([x['common_suffix_kl'] for x in b])))


if __name__=='__main__':run()

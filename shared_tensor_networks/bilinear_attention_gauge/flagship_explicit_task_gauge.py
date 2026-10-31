"""Exact QK1-key gauges from competent explicit forced-choice behaviors."""

import argparse
import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_screen import TOKENIZER, examples
from flagship_full_context_sensitivity_gauge import SensitivityEdit, normalize_rows
from flagship_full_model_eval import GPT, GPTConfig, logits
from flagship_matrix_free import HD, SNAPSHOT
from flagship_shared_key_gauge_priors import varimax
from flagship_transfer import ART


FAMILIES=('subject_verb_agreement','number_sequence','bracket_closure',
          'antonym','country_capital')
FIT_FAMILIES=FAMILIES[:3]
HELD_FAMILIES=FAMILIES[3:]
BASIS_NAME='all_head_function_aware_shared_key_s642_n32768_t160.npz'
PRIOR_NAME='all_head_shared_key_gauge_priors_function_n2048_p2048_i120_d600.npz'
FULL_CONTEXT_NAME='all_head_full_context_sensitivity_gauge_train64_test24_l64_s1031.npz'
STEM='all_head_explicit_task_gauge_train9811_val9911_test10011'


def make_rows(tokenizer,seed,n):
    rows=[]
    for family,prompt,correct,foil in examples(seed,n):
        if family not in FAMILIES:continue
        correct_ids=tokenizer.encode(correct,add_special_tokens=False).ids
        foil_ids=tokenizer.encode(foil,add_special_tokens=False).ids
        assert len(correct_ids)==len(foil_ids)==1
        rows.append({'family':family,'prompt':prompt,
                     'prompt_ids':tokenizer.encode(prompt,add_special_tokens=False).ids,
                     'correct_id':correct_ids[0],'foil_id':foil_ids[0]})
    return rows


def panel(model,editor,basis,rows,max_length):
    gradients=np.zeros((len(rows),max_length,256),dtype=np.float32)
    lengths=[];native_margins=[];cut_margins=[];gate=None
    basis_t=torch.from_numpy(basis.astype(np.float32))
    for i,row in enumerate(rows):
        idx=torch.tensor(row['prompt_ids'],dtype=torch.long)[None,:]
        editor.reset()
        with torch.no_grad():
            native=logits(model,idx)[0,-1].float()
        native_margin=float(native[row['correct_id']]-native[row['foil_id']])
        editor.set_shared(basis)
        value=logits(model,idx)[0,-1].float()
        margin=value[row['correct_id']]-value[row['foil_id']]
        grad=torch.autograd.grad(margin,editor.last_pre)[0][0]
        projected=(grad.reshape(len(row['prompt_ids']),-1)@basis_t).detach().numpy()
        assert np.isfinite(projected).all()
        gradients[i,:len(projected)]=projected
        lengths.append(len(projected));native_margins.append(native_margin)
        cut_margins.append(float(margin.detach()))
        if gate is None:
            position=len(projected)-1
            axis=int(np.argmax(np.abs(projected[position])))
            editor.injection=(position,basis_t[:,axis],.1)
            plus=logits(model,idx)[0,-1].float()
            plus=plus[row['correct_id']]-plus[row['foil_id']]
            editor.injection=(position,basis_t[:,axis],-.1)
            minus=logits(model,idx)[0,-1].float()
            minus=minus[row['correct_id']]-minus[row['foil_id']]
            editor.injection=None
            finite=float(((plus-minus)/.2).detach())
            analytic=float(projected[position,axis])
            gate={'position':position,'axis':axis,
                  'analytic':analytic,'central_difference':finite,
                  'absolute_error':abs(analytic-finite)}
            assert gate['absolute_error']<max(2e-4,.05*abs(analytic)),gate
        if (i+1)%40==0:print('panel',i+1,len(rows),flush=True)
    return {'gradients':gradients,
            'lengths':np.asarray(lengths,dtype=np.int16),
            'family':np.asarray([FAMILIES.index(row['family']) for row in rows],dtype=np.int8),
            'native_margin':np.asarray(native_margins,dtype=np.float32),
            'cut_margin':np.asarray(cut_margins,dtype=np.float32),
            'finite_difference_gate':gate}


def collect():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    rows={'train':make_rows(tokenizer,9811,40),
          'validation':make_rows(tokenizer,9911,20),
          'test':make_rows(tokenizer,10011,40)}
    max_length=max(len(row['prompt_ids']) for panel_rows in rows.values()
                   for row in panel_rows)
    basis=np.load(ART/BASIS_NAME)['basis']
    assert basis.shape==(9*HD,256)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    editor=SensitivityEdit(model,max_length,basis)
    panels={}
    for name,panel_rows in rows.items():
        panels[name]=panel(model,editor,basis,panel_rows,max_length)
        print(name,len(panel_rows),flush=True)
    np.savez_compressed(ART/f'{STEM}.npz',
        **{f'{name}_{field}':value for name,data in panels.items()
           for field,value in data.items() if isinstance(value,np.ndarray)})
    metadata={'stem':STEM,'families':FAMILIES,'fit_families':FIT_FAMILIES,
              'held_families':HELD_FAMILIES,'max_prompt_length':max_length,
              'basis':BASIS_NAME,'rows':rows,
              'head_equivalence_max_abs':editor.head_equivalence_max_abs,
              'finite_difference_gates':{name:data['finite_difference_gate']
                                         for name,data in panels.items()},
              'scope':'native versus fixed-rank256 18-layer forced-choice margin, gradients of cut margin w.r.t first-layer QK1 keys'}
    (ART/f'{STEM}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def effective_axes(gradient,length,rotation):
    coord=gradient[:length]@rotation
    energy=np.sum(coord*coord,axis=0)
    return float(np.sum(np.sqrt(energy))**2/max(np.sum(energy),1e-30))


def fit(stem=STEM):
    metadata=json.loads((ART/f'{stem}.json').read_text())
    arrays=np.load(ART/f'{stem}.npz')
    gradients=arrays['train_gradients'].astype(np.float64)
    lengths=arrays['train_lengths']
    family=arrays['train_family']
    assert np.all(lengths>0)
    fit_examples=np.isin(family,[FAMILIES.index(x) for x in FIT_FAMILIES])
    def normalized_rows(selected):
        x=np.concatenate([gradients[i,:lengths[i]] for i in np.where(selected)[0]])
        normalized,floor=normalize_rows(x)
        weights=np.concatenate([np.full(lengths[i],1/np.sqrt(lengths[i]))
                                for i in np.where(selected)[0]])
        return normalized*weights[:,None],floor
    fit_rows,fit_floor=normalized_rows(fit_examples)
    all_rows,all_floor=normalized_rows(np.ones(len(family),dtype=bool))
    fitted,fit_trace=varimax(fit_rows,np.eye(256),max_iter=240)
    oracle,oracle_trace=varimax(all_rows,np.eye(256),max_iter=240)
    plus_rotations={};plus_traces={};plus_floors={}
    for added in HELD_FAMILIES:
        selected=np.isin(family,[FAMILIES.index(x) for x in (*FIT_FAMILIES,added)])
        rows,floor=normalized_rows(selected)
        rotation,trace=varimax(rows,np.eye(256),max_iter=240)
        name=f'fit_plus_{added}'
        plus_rotations[name]=rotation
        plus_traces[name]=trace
        plus_floors[name]=floor
    prior=np.load(ART/PRIOR_NAME)
    context=np.load(ART/FULL_CONTEXT_NAME)
    choices={'native':np.eye(256),'weight_0':prior['weight_0'],
             'response_0':prior['response_0'],
             'corpus_full_context':context['rotation_full_context_varimax'],
             'explicit_fit_families':fitted,
             **plus_rotations,
             'all_families_oracle':oracle}
    scores={}
    for name,rotation in choices.items():
        assert np.max(np.abs(rotation.T@rotation-np.eye(256)))<1e-10
        scores[name]={}
        for panel_name in ('train','validation','test'):
            g=arrays[f'{panel_name}_gradients']
            lens=arrays[f'{panel_name}_lengths']
            fam=arrays[f'{panel_name}_family']
            values=np.array([effective_axes(g[i],int(lens[i]),rotation)
                             for i in range(len(g))])
            scores[name][panel_name]={family_name:{
                'n':int(np.sum(fam==index)),
                'mean_effective_axes':float(values[fam==index].mean())}
                for index,family_name in enumerate(FAMILIES)}
    metadata['gauge_fit']={'fit_examples':int(fit_examples.sum()),
        'fit_family_indices':[FAMILIES.index(x) for x in FIT_FAMILIES],
        'normalization_floor':fit_floor,
        'oracle_normalization_floor':all_floor,
        'plus_family_normalization_floors':plus_floors,
        'plus_family_traces':plus_traces,
        'fit_trace':fit_trace,'oracle_trace':oracle_trace,
        'scores':scores,
        'caveat':'response-gradient sparsity is not finite causal fidelity; oracle saw held task families'}
    (ART/f'{stem}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    np.savez_compressed(ART/f'{stem}_rotations.npz',**choices)
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fit-stem')
    args=parser.parse_args()
    if args.fit_stem:fit(args.fit_stem)
    else:collect()

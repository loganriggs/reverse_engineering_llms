"""Add a training-dose-matched pair-response axis to same-position panel."""

import argparse
import json
import math

import numpy as np
import torch

from flagship_component_labels import Decoder
from flagship_full_context_finite_axis_test import (
    STEM,OneKeyAxisDeletion,axis_profiles,logprob_last,
)
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART


SOURCE='all_head_full_context_same_position_n24_s1231_l64_f100.json'
NAME='pair_response_punctuation_axis_matched'
AXIS=16


def run(source=SOURCE):
    torch.set_num_threads(4)
    panel=json.loads((ART/source).read_text())
    assert panel['length']==64 and panel['n_docs_per_corpus']==24
    if NAME in panel['detail']:raise ValueError('control already present')
    basis=np.load(ART/'all_head_function_aware_shared_key_s642_n32768_t160.npz')['basis'].astype(np.float64)
    rotation=np.load(ART/'all_head_shared_key_gauge_priors_function_n2048_p2048_i120_d600.npz')['response_0']
    train=np.load(ART/f'{STEM}.npz')
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    profile=axis_profiles(state,basis,train,{'pair_response':rotation},Decoder())['pair_response'][AXIS]
    target=panel['detail']['punctuation_axis_1']['energy_share']
    fraction=math.sqrt(target/profile['energy_share'])
    assert 0<fraction<1
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval();model.load_state_dict(state)
    del state
    editor=OneKeyAxisDeletion(model,panel['length'])
    u=(basis@rotation).astype(np.float32)
    gate=0.;records=[]
    corpora={'fineweb_test':CORPUS,**DATASETS}
    with torch.no_grad():
        for corpus_name,path in corpora.items():
            corpus=np.load(path,mmap_mode='r')
            selected=[r for r in panel['records']
                      if r['corpus']==corpus_name and r['method']=='punctuation_axis_1']
            assert len(selected)==2*panel['n_docs_per_corpus']
            for previous in selected:
                tokens=np.asarray(corpus[previous['document_id'],
                    previous['start']:previous['start']+panel['length']+1],dtype=np.int64).copy()
                position=previous['key_position']
                tokens[position]=previous['key_id']
                idx=torch.from_numpy(tokens[:-1].copy())[None,:]
                target_id=int(tokens[-1]);assert target_id==previous['target_id']
                editor.axis=None;editor.set_shared(u)
                reference=logits(model,idx)
                head=editor.last_attention.clone()
                reference_lp=logprob_last(reference,target_id)
                gate=max(gate,abs(reference_lp-previous['common_baseline_logprob']))
                editor.axis=AXIS;editor.key_position=position;editor.fraction=fraction
                edited=logits(model,idx)
                change=logprob_last(edited,target_id)-reference_lp
                dose=float((editor.last_attention[:, -1]-head[:, -1]).square().sum())
                full_dose=float((editor.last_attention-head).square().sum())
                records.append(dict(corpus=corpus_name,
                    document_id=previous['document_id'],start=previous['start'],
                    target_id=target_id,method=NAME,
                    key_category=previous['key_category'],key_position=position,
                    key_id=previous['key_id'],original_key_id=previous['original_key_id'],
                    position_gap=0,baseline_logprob=reference_lp,
                    common_baseline_logprob=previous['common_baseline_logprob'],
                    native_logprob=previous['native_logprob'],
                    logprob_change=change,logprob_change_sq=change**2,
                    full_logit_change_sq=float((edited[0,-1]-reference[0,-1]).square().sum()),
                    head_output_change_sq=dose,
                    head_output_full_sequence_change_sq=full_dose))
            print(corpus_name,'variants',len(selected),flush=True)
    assert gate<1e-3,gate
    panel['detail'][NAME]=dict(gauge='pair_response',**profile,
        deletion_fraction=fraction,
        matched_to='punctuation_axis_1',
        matching='predicted training deletion-effect energy via squared fraction',
        previous_pair_label='space_word -> punctuation')
    panel['records'].extend(records)
    panel['pair_response_rotation_max_abs_target_logprob']=gate
    panel['pair_response_control_source']='all_head_shared_key_gauge_priors_function_n2048_p2048_i120_d600.npz'
    output=source.replace('.json','_with_pair_control.json')
    (ART/output).write_text(json.dumps(panel,indent=2)+'\n')
    return panel


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default=SOURCE)
    a=p.parse_args();run(a.source)

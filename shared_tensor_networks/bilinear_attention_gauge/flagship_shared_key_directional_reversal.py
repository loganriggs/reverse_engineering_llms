"""Four-condition query/key token-form test of two exact-gauge axes."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from flagship_component_labels import CAT_INDEX,Decoder,check_decoder
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_shared_key_counterfactual_causality import CaptureEdit,norm2,windows
from flagship_shared_key_input_edit_fidelity import kl
from flagship_transfer import ART


FORWARD='response_0_c16_matched'
REVERSE='same_gauge_response_0_c165'
CONDITIONS=('word_query_word_key','word_query_punct_key',
            'punct_query_word_key','punct_query_punct_key')


def cuts(basis,rotations):
    forward_info=json.loads((ART/'all_head_shared_key_counterfactual_causality_n8_s91-97.json').read_text())
    reverse_info=json.loads((ART/'all_head_shared_key_same_gauge_controls_n96.json').read_text())
    dose_forward=forward_info['detail'][FORWARD]['dose']
    dose_reverse=reverse_info['detail'][REVERSE]['dose']
    methods={};detail={}
    u=basis@rotations['response_0']
    for name,component,dose,label in (
        ('forward_c16',16,dose_forward,'space_word -> punctuation'),
        ('reverse_c165',165,dose_reverse,'punctuation -> space_word')):
        weighted=u.copy();weighted[:,component]*=np.sqrt(1-dose)
        methods[name]=weighted
        detail[name]=dict(component=component,dose=dose,
            proposed_label=label,kind='candidate',
            matched_validation_pair_energy=float(
                forward_info['detail'][FORWARD]['matched_selection_pair_energy']))
    random=basis@rotations['random_0']
    for component in (94,106):
        name=f'random_c{component}'
        methods[name]=np.delete(random,component,axis=1)
        detail[name]=dict(component=component,dose=1.,
                          kind='random_control')
    return methods,detail


def run(n_docs=8,seeds=(111,117),batch_size=2,length=128):
    torch.set_num_threads(4)
    decoder=Decoder();check_decoder(decoder)
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    basis=np.load(ART/'all_head_function_aware_shared_key_s642_n32768_t160.npz')[
          'basis'].astype(np.float64)
    rotations=np.load(ART/'all_head_shared_key_gauge_priors_function_n2048_p2048_i120_d600.npz')
    methods,detail=cuts(basis,rotations)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval();model.load_state_dict(state);del state
    editor=CaptureEdit(model,length)
    excluded={}
    prior_files=list(ART.glob('all_head_shared_key_full_model_n16*.json'))+[
        ART/'all_head_shared_key_counterfactual_causality_n8_s91-97.json',
        ART/'all_head_shared_key_counterfactual_causality_n8_s101-107_punct25-26_word286-257.json']
    for path in prior_files:
        prior=json.loads(path.read_text())
        for row in prior['records']:
            excluded.setdefault(row['corpus'],set()).add(row['document_id'])
    corpora={'fineweb_test':(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[];key_edit_responses=[];fidelity=[];identity_gate=None;skipped={}
    with torch.no_grad():
        for corpus_name,(path,begin,end) in corpora.items():
            corpus=np.load(path,mmap_mode='r')
            used=excluded.get(corpus_name,set()).copy()
            for seed in seeds:
                entries,missed=windows(corpus,begin,end,seed,n_docs,length,
                    decoder,used,(11,26),(262,290))
                skipped[f'{corpus_name}_{seed}']=missed
                used.update(entry['document_id'] for entry in entries)
                for offset in range(0,len(entries),batch_size):
                    chunk=entries[offset:offset+batch_size]
                    variants=[];meta=[]
                    for local,entry in enumerate(chunk):
                        original=entry['variants'][0]
                        j=entry['key'];i=entry['query']
                        query_punct=(13,25)[(offset+local)%2]
                        key_punct=(11,26)[(offset+local)%2]
                        assert decoder.category(original[j])==CAT_INDEX['space_word']
                        assert decoder.category(original[i])==CAT_INDEX['space_word']
                        quad=np.repeat(original[None,:],4,axis=0)
                        quad[1,j]=key_punct
                        quad[2,i]=query_punct
                        quad[3,j]=key_punct;quad[3,i]=query_punct
                        variants.append(quad)
                        meta.append(dict(corpus=corpus_name,seed=seed,
                            document_id=entry['document_id'],start=entry['start'],
                            query_position=i,key_position=j,
                            original_query_id=int(original[i]),
                            original_key_id=int(original[j]),
                            query_punctuation_id=query_punct,
                            key_punctuation_id=key_punct))
                    idx=torch.from_numpy(np.concatenate(variants).copy())
                    editor.reset();native=logits(model,idx).numpy()
                    if identity_gate is None:
                        editor.set_shared(np.eye(9*128,dtype=np.float32))
                        identity_gate=float(np.max(np.abs(logits(model,idx).numpy()-native)))
                        assert identity_gate<1e-3,identity_gate
                    editor.set_shared(basis)
                    reference=logits(model,idx).numpy()
                    reference_head=editor.last_attention.numpy()
                    for local,item in enumerate(meta):
                        b=4*local;i=item['query_position']
                        rows={}
                        for condition in range(4):
                            rows[CONDITIONS[condition]]=kl(
                                native[b+condition,i],reference[b+condition,i])
                        interaction_native=(native[b+3,i]-native[b+2,i]-
                                            native[b+1,i]+native[b,i])
                        interaction_reduced=(reference[b+3,i]-reference[b+2,i]-
                                             reference[b+1,i]+reference[b,i])
                        key_edit_fidelity={}
                        for query_form,lo,hi in (('space_word',0,1),
                                                 ('punctuation',2,3)):
                            native_delta=native[b+hi,i]-native[b+lo,i]
                            reduced_delta=reference[b+hi,i]-reference[b+lo,i]
                            key_edit_fidelity[query_form]=dict(
                                native_energy=norm2(native_delta),
                                error_energy=norm2(native_delta-reduced_delta))
                        fidelity.append(dict(**item,
                            native_to_reduced_query_kl=rows,
                            native_to_reduced_key_edit_response=key_edit_fidelity,
                            native_interaction_energy=norm2(interaction_native),
                            interaction_error_energy=norm2(
                                interaction_native-interaction_reduced)))
                    for name,cut in methods.items():
                        editor.set_shared(cut)
                        candidate=logits(model,idx).numpy()
                        candidate_head=editor.last_attention.numpy()
                        for local,item in enumerate(meta):
                            b=4*local;i=item['query_position']
                            for query_form,lo,hi in (('space_word',0,1),
                                                     ('punctuation',2,3)):
                                baseline_delta=(reference[b+hi,i]-
                                                reference[b+lo,i])
                                candidate_delta=(candidate[b+hi,i]-
                                                 candidate[b+lo,i])
                                baseline_head_delta=(reference_head[b+hi,:,i]-
                                                     reference_head[b+lo,:,i])
                                candidate_head_delta=(candidate_head[b+hi,:,i]-
                                                      candidate_head[b+lo,:,i])
                                key_edit_responses.append(dict(
                                    **item,method=name,query_form=query_form,
                                    baseline_logit_response_energy=norm2(baseline_delta),
                                    component_logit_response_energy=norm2(
                                        baseline_delta-candidate_delta),
                                    baseline_head_response_energy=norm2(baseline_head_delta),
                                    component_head_response_energy=norm2(
                                        baseline_head_delta-candidate_head_delta)))
                            for condition,label in enumerate(CONDITIONS):
                                ref=reference[b+condition,i]
                                edit=candidate[b+condition,i]
                                head_ref=reference_head[b+condition,:,i]
                                head_edit=candidate_head[b+condition,:,i]
                                records.append(dict(**item,method=name,
                                    condition=label,query_kl=kl(ref,edit),
                                    logit_deletion_energy=norm2(ref-edit),
                                    head_deletion_energy=norm2(head_ref-head_edit)))
                print(corpus_name,seed,flush=True)
    result=dict(layer=0,rank=256,length=length,seeds=list(seeds),
        n_docs_per_seed=n_docs,batch_size=batch_size,
        conditions=list(CONDITIONS),
        query_punctuation_ids=[13,25],key_punctuation_ids=[11,26],
        candidate_selection='two previously chosen, validation-energy-matched response-varimax axes with opposite proposed query/key token-form labels',
        excluded_document_artifacts=[p.name for p in prior_files],
        skipped_windows=skipped,identity_projection_max_abs_logit=identity_gate,
        detail=detail,fidelity=fidelity,records=records,
        key_edit_responses=key_edit_responses,
        scope='full-model directional input factorial on new documents; still an edit of a chosen reduced circuit, not physical-node recovery')
    stem=(f'all_head_shared_key_directional_reversal_n{n_docs}_s'+
          '-'.join(map(str,seeds)))
    (ART/f'{stem}.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n-docs',type=int,default=8)
    p.add_argument('--seeds',type=int,nargs='+',default=[111,117])
    p.add_argument('--batch-size',type=int,default=2)
    a=p.parse_args()
    run(a.n_docs,tuple(a.seeds),a.batch_size)

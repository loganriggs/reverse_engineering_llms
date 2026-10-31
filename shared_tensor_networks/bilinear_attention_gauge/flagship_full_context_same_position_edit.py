"""Same-position punctuation versus word-token edits for selected QK1 axes."""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_component_labels import CAT_INDEX,Decoder
from flagship_full_context_finite_axis_test import (
    STEM,OneKeyAxisDeletion,logprob_last,select_methods,
)
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART


PUNCTUATION=(11,13,25,26)
SPACE_WORD=(262,290,286,257)


def excluded_documents(corpus_name,prior):
    key='test_fineweb' if corpus_name=='fineweb_test' else f'test_{corpus_name}'
    excluded=set(prior['corpus_splits'][key])
    natural=ART/'all_head_full_context_finite_axis_n24_s1131_l64_f100.json'
    if natural.exists():
        panel=json.loads(natural.read_text())
        excluded.update(r['document_id'] for r in panel['records']
                        if r['corpus']==corpus_name)
    return excluded


def windows(corpus,begin,end,excluded,n_docs,seed,length):
    rng=np.random.default_rng(seed)
    choices=np.setdiff1d(np.arange(begin,end),np.array(sorted(excluded),dtype=int))
    docs=rng.choice(choices,size=n_docs,replace=False)
    out=[]
    for index,doc in enumerate(docs):
        start=int(rng.integers(0,corpus.shape[1]-length-1))
        original=np.asarray(corpus[doc,start:start+length+1],dtype=np.int64)
        position=int(rng.integers(length-24,length-8))
        punct=PUNCTUATION[index%len(PUNCTUATION)]
        word=SPACE_WORD[index%len(SPACE_WORD)]
        variants={'punctuation':original.copy(),'space_word':original.copy()}
        variants['punctuation'][position]=punct
        variants['space_word'][position]=word
        out.append(dict(document_id=int(doc),start=start,position=position,
            original_key_id=int(original[position]),punctuation_id=punct,
            word_id=word,variants=variants))
    return out


def run(n_docs=24,seed=1231,length=64,fraction=1.):
    torch.set_num_threads(4)
    assert length==64 and 0<fraction<=1
    decoder=Decoder()
    assert all(decoder.category(t)==CAT_INDEX['punctuation'] for t in PUNCTUATION)
    assert all(decoder.category(t)==CAT_INDEX['space_word'] for t in SPACE_WORD)
    prior=json.loads((ART/f'{STEM}.json').read_text())
    train=np.load(ART/f'{STEM}.npz')
    basis=np.load(ART/'all_head_function_aware_shared_key_s642_n32768_t160.npz')['basis'].astype(np.float64)
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    rotations,methods,detail=select_methods(state,basis,train,decoder)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval();model.load_state_dict(state)
    del state
    editor=OneKeyAxisDeletion(model,length)
    corpora={'fineweb_test':(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[];coverage={};gate=0.;gauge_gate=0.;zero_gate=0.
    with torch.no_grad():
        for corpus_index,(corpus_name,(path,begin,end)) in enumerate(corpora.items()):
            corpus=np.load(path,mmap_mode='r')
            entries=windows(corpus,begin,end,excluded_documents(corpus_name,prior),
                            n_docs,seed+100000*corpus_index,length)
            coverage[corpus_name]=dict(selected_documents=len(entries),
                selected_ids=[entry['document_id'] for entry in entries])
            for entry in entries:
                for category,tokens in entry['variants'].items():
                    idx=torch.from_numpy(tokens[:-1].copy())[None,:]
                    target=int(tokens[-1])
                    editor.axis=None;editor.reset()
                    native=logits(model,idx)
                    editor.set_shared(basis)
                    baseline=logits(model,idx)
                    native_lp=logprob_last(native,target)
                    baseline_lp=logprob_last(baseline,target)
                    if len(records)<2*len(methods):
                        editor.set_shared(np.eye(9*HD,dtype=np.float32))
                        gate=max(gate,float((logits(model,idx)-native).abs().max()))
                    for name,(gauge,axis) in methods.items():
                        editor.set_shared((basis@rotations[gauge]).astype(np.float32))
                        editor.axis=None
                        gauge_baseline=logits(model,idx)
                        head=editor.last_attention.clone()
                        gauge_lp=logprob_last(gauge_baseline,target)
                        gauge_gate=max(gauge_gate,float((gauge_baseline-baseline).abs().max()))
                        editor.axis=axis;editor.key_position=entry['position']
                        if len(records)<2*len(methods):
                            editor.fraction=0.
                            zero=logits(model,idx)
                            zero_gate=max(zero_gate,float((zero-gauge_baseline).abs().max()))
                        editor.fraction=fraction
                        edited=logits(model,idx)
                        change=logprob_last(edited,target)-gauge_lp
                        logit_change_sq=float((edited[0,-1]-
                            gauge_baseline[0,-1]).square().sum())
                        dose=float((editor.last_attention[:, -1]-head[:, -1]).square().sum())
                        full_dose=float((editor.last_attention-head).square().sum())
                        records.append(dict(corpus=corpus_name,
                            document_id=entry['document_id'],start=entry['start'],
                            target_id=target,method=name,key_category=category,
                            key_position=entry['position'],key_id=int(tokens[entry['position']]),
                            original_key_id=entry['original_key_id'],
                            position_gap=0,baseline_logprob=gauge_lp,
                            common_baseline_logprob=baseline_lp,native_logprob=native_lp,
                            logprob_change=change,logprob_change_sq=change**2,
                            full_logit_change_sq=logit_change_sq,
                            head_output_change_sq=dose,
                            head_output_full_sequence_change_sq=full_dose))
                        editor.axis=None
            print(corpus_name,'docs',len(entries),flush=True)
    assert gate<1e-3 and gauge_gate<1e-3 and zero_gate<1e-3
    output=dict(layer=0,heads=list(range(9)),rank=256,length=length,
        n_docs_per_corpus=n_docs,seed=seed,deletion_fraction=fraction,
        selection='same training-only selected axes as natural-key panel',
        intervention='two alternate key token IDs at exactly the same key position; delete one QK1-key coordinate in the fixed rank-256 model',
        punctuation_ids=list(PUNCTUATION),space_word_ids=list(SPACE_WORD),
        readout='last-position next-token log probability after all 18 layers',
        identity_max_abs_logit=gate,gauge_rotation_max_abs_logit=gauge_gate,
        zero_deletion_max_abs_logit=zero_gate,detail=detail,coverage=coverage,
        records=records,
        caveat='controlled token substitution changes other first-layer paths too; deletion isolates the chosen reduced QK1-key coordinate only within each variant')
    stem=f'all_head_full_context_same_position_n{n_docs}_s{seed}_l{length}_f{round(fraction*100)}'
    (ART/f'{stem}.json').write_text(json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n-docs',type=int,default=24)
    p.add_argument('--seed',type=int,default=1231)
    p.add_argument('--length',type=int,default=64)
    p.add_argument('--fraction',type=float,default=1.)
    a=p.parse_args()
    run(a.n_docs,a.seed,a.length,a.fraction)

"""Discover a downstream output signature on validation contexts and four token IDs.

The other four punctuation/word token pairs and all test-corpus documents are
reserved for a subsequent frozen-hypothesis validation.
"""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_component_labels import CATEGORIES,Decoder
from flagship_full_context_finite_axis_test import STEM,OneKeyAxisDeletion
from flagship_full_context_same_position_edit import unseen_frequency_pairs,windows
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_transfer import ART


PAIR_SOURCE='all_head_full_context_same_position_n24_s1331_l64_f100_unseen_with_pair_control.json'


def category_index(decoder,vocab):
    return torch.tensor([decoder.category(i) for i in range(vocab)],dtype=torch.int64)


def category_mass(output,index):
    probability=F.softmax(output[0,-1].to(torch.float64),dim=-1)
    return torch.bincount(index,weights=probability,minlength=len(CATEGORIES)).numpy()


def run(n_docs=64,seed=1431,length=64):
    torch.set_num_threads(4)
    assert n_docs%4==0 and length==64
    decoder=Decoder()
    prior=json.loads((ART/PAIR_SOURCE).read_text())
    pairs=prior['token_pairs']
    discovery=pairs[::2];held_out=pairs[1::2]
    sensitivity=json.loads((ART/f'{STEM}.json').read_text())
    excluded=set(sensitivity['corpus_splits']['validation_fineweb'])
    corpus=np.load(CORPUS,mmap_mode='r')
    entries=windows(corpus,15000,17500,excluded,n_docs,seed,length,
        tuple(p['punctuation_id'] for p in discovery),
        tuple(p['word_id'] for p in discovery))
    basis=np.load(ART/'all_head_function_aware_shared_key_s642_n32768_t160.npz')['basis'].astype(np.float64)
    starts=np.load(ART/f'{STEM}_independent_starts.npz')
    old=np.load(ART/'all_head_shared_key_gauge_priors_function_n2048_p2048_i120_d600.npz')
    methods={
        'full_context_axis_7':dict(basis=(basis@starts['full_context_varimax']).astype(np.float32),
                                   axis=prior['detail']['punctuation_axis_1']['axis'],fraction=1.),
        'pair_response_axis_16':dict(basis=(basis@old['response_0']).astype(np.float32),
                                     axis=prior['detail']['pair_response_punctuation_axis_matched']['axis'],
                                     fraction=prior['detail']['pair_response_punctuation_axis_matched']['deletion_fraction']),
    }
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    editor=OneKeyAxisDeletion(model,length)
    categories=category_index(decoder,model.lm_head.weight.shape[0])
    arrays={name:np.empty((n_docs,2,len(categories)),dtype=np.float16)
            for name in methods}
    mass={name:np.empty((n_docs,2,len(CATEGORIES)),dtype=np.float64)
          for name in methods}
    records=[];gauge_gate=0.
    with torch.no_grad():
        for doc_index,entry in enumerate(entries):
            for variant_index,(variant,tokens) in enumerate(entry['variants'].items()):
                idx=torch.from_numpy(tokens[:-1].copy())[None,:]
                editor.axis=None;editor.set_shared(basis)
                common=logits(model,idx)
                for name,method in methods.items():
                    editor.axis=None;editor.set_shared(method['basis'])
                    baseline=logits(model,idx)
                    gauge_gate=max(gauge_gate,float((baseline-common).abs().max()))
                    editor.axis=method['axis'];editor.key_position=entry['position']
                    editor.fraction=method['fraction']
                    edited=logits(model,idx)
                    delta=(edited[0,-1]-baseline[0,-1]).to(torch.float32)
                    arrays[name][doc_index,variant_index]=delta.numpy().astype(np.float16)
                    mass[name][doc_index,variant_index]=(
                        category_mass(edited,categories)-category_mass(baseline,categories))
                    editor.axis=None
            records.append(dict(document_id=entry['document_id'],start=entry['start'],
                key_position=entry['position'],punctuation_id=entry['punctuation_id'],
                word_id=entry['word_id']))
            if (doc_index+1)%16==0:print('discovery docs',doc_index+1,flush=True)
    assert gauge_gate<1e-3,gauge_gate
    summaries={};saved={}
    for name,delta in arrays.items():
        d=delta[:,0].astype(np.float64)-delta[:,1].astype(np.float64)
        mean=d.mean(0)
        coherence=float(np.sum(mean**2)/np.mean(np.sum(d*d,axis=1)))
        norm=np.linalg.norm(d,axis=1)
        projection=d@mean/np.maximum(norm*np.linalg.norm(mean),1e-30)
        positive=np.argsort(mean)[-20:][::-1]
        negative=np.argsort(mean)[:20]
        category_interaction=(mass[name][:,0]-mass[name][:,1]).mean(0)
        summaries[name]=dict(
            response_coherence=coherence,
            median_document_cosine_to_mean=float(np.median(projection)),
            positive_cosine_documents=int(np.sum(projection>0)),
            top_positive_output_tokens=[dict(id=int(i),text=decoder.decode(i),
                mean_logit_interaction=float(mean[i])) for i in positive],
            top_negative_output_tokens=[dict(id=int(i),text=decoder.decode(i),
                mean_logit_interaction=float(mean[i])) for i in negative],
            category_probability_mass_interaction={label:float(value)
                for label,value in zip(CATEGORIES,category_interaction)})
        saved[f'{name}_delta']=delta
        saved[f'{name}_mean_interaction']=mean.astype(np.float32)
    output=dict(layer=0,rank=256,length=length,seed=seed,n_docs=n_docs,
        split='FineWeb validation documents 15000:17500, excluding sensitivity-gauge validation documents',
        readout='final-position full-vocabulary logit deletion response; difference of punctuation and matched-word variants',
        discovery_pairs=discovery,held_out_test_pairs=held_out,
        gauge_rotation_max_abs_logit=gauge_gate,
        methods={name:dict(axis=method['axis'],fraction=method['fraction'])
                 for name,method in methods.items()},
        records=records,summaries=summaries)
    stem=f'all_head_output_signature_discovery_n{n_docs}_s{seed}_l{length}'
    (ART/f'{stem}.json').write_text(json.dumps(output,indent=2)+'\n')
    np.savez_compressed(ART/f'{stem}.npz',**saved)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n-docs',type=int,default=64)
    p.add_argument('--seed',type=int,default=1431)
    a=p.parse_args();run(a.n_docs,a.seed)

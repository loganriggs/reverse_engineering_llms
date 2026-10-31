"""Frozen output-signature validation on four unseen token pairs and three corpora."""

import argparse
import json

import numpy as np
import torch

from flagship_component_labels import CATEGORIES,Decoder
from flagship_full_context_finite_axis_test import STEM,OneKeyAxisDeletion
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_output_signature_discovery import (
    PAIR_SOURCE,category_index,category_mass,
)
from flagship_transfer import ART


DISCOVERY='all_head_output_signature_discovery_n64_s1431_l64'


def run(discovery_stem=DISCOVERY):
    torch.set_num_threads(4)
    discovery=json.loads((ART/f'{discovery_stem}.json').read_text())
    source=np.load(ART/f'{discovery_stem}.npz')
    panel=json.loads((ART/PAIR_SOURCE).read_text())
    held_ids={p['punctuation_id'] for p in discovery['held_out_test_pairs']}
    assert held_ids=={p['punctuation_id'] for p in panel['token_pairs'][1::2]}
    basis=np.load(ART/'all_head_function_aware_shared_key_s642_n32768_t160.npz')['basis'].astype(np.float64)
    starts=np.load(ART/f'{STEM}_independent_starts.npz')
    old=np.load(ART/'all_head_shared_key_gauge_priors_function_n2048_p2048_i120_d600.npz')
    methods={
        'full_context_axis_7':dict(basis=(basis@starts['full_context_varimax']).astype(np.float32),
            axis=discovery['methods']['full_context_axis_7']['axis'],
            fraction=discovery['methods']['full_context_axis_7']['fraction']),
        'pair_response_axis_16':dict(basis=(basis@old['response_0']).astype(np.float32),
            axis=discovery['methods']['pair_response_axis_16']['axis'],
            fraction=discovery['methods']['pair_response_axis_16']['fraction']),
    }
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    decoder=Decoder();categories=category_index(decoder,model.lm_head.weight.shape[0])
    editor=OneKeyAxisDeletion(model,panel['length'])
    corpora={'fineweb_test':CORPUS,**DATASETS}
    records=[];gauge_gate=0.
    with torch.no_grad():
        for corpus_name,path in corpora.items():
            corpus=np.load(path,mmap_mode='r')
            selected=[r for r in panel['records']
                if r['corpus']==corpus_name and r['method']=='punctuation_axis_1'
                and r['key_category']=='punctuation' and r['key_id'] in held_ids]
            assert len(selected)==12,len(selected)
            for record in selected:
                delta={};mass={}
                for variant in ('punctuation','space_word'):
                    matching=next(r for r in panel['records']
                        if r['corpus']==corpus_name and r['method']=='punctuation_axis_1'
                        and r['document_id']==record['document_id']
                        and r['key_category']==variant)
                    tokens=np.asarray(corpus[record['document_id'],
                        record['start']:record['start']+panel['length']+1],dtype=np.int64).copy()
                    tokens[record['key_position']]=matching['key_id']
                    idx=torch.from_numpy(tokens[:-1].copy())[None,:]
                    editor.axis=None;editor.set_shared(basis)
                    common=logits(model,idx)
                    for name,method in methods.items():
                        editor.axis=None;editor.set_shared(method['basis'])
                        baseline=logits(model,idx)
                        gauge_gate=max(gauge_gate,float((baseline-common).abs().max()))
                        editor.axis=method['axis'];editor.key_position=record['key_position']
                        editor.fraction=method['fraction']
                        edited=logits(model,idx)
                        delta.setdefault(name,{})[variant]=(
                            edited[0,-1]-baseline[0,-1]).numpy().astype(np.float64)
                        mass.setdefault(name,{})[variant]=(
                            category_mass(edited,categories)-category_mass(baseline,categories))
                        editor.axis=None
                for name in methods:
                    d=delta[name]['punctuation']-delta[name]['space_word']
                    mean=source[f'{name}_mean_interaction'].astype(np.float64)
                    category_d=mass[name]['punctuation']-mass[name]['space_word']
                    contrast=float(category_d[CATEGORIES.index('punctuation')]-
                                   category_d[CATEGORIES.index('space_word')])
                    expected=discovery['summaries'][name]['category_probability_mass_interaction']
                    expected_contrast=expected['punctuation']-expected['space_word']
                    records.append(dict(corpus=corpus_name,document_id=record['document_id'],
                        punctuation_id=record['key_id'],method=name,
                        response_norm=float(np.linalg.norm(d)),
                        cosine_to_discovery_mean=float(d@mean/
                            max(np.linalg.norm(d)*np.linalg.norm(mean),1e-30)),
                        projection_onto_discovery_mean=float(d@mean/
                            max(mean@mean,1e-30)),
                        output_category_contrast=contrast,
                        expected_discovery_contrast=float(expected_contrast),
                        output_category_contrast_sign_matches=bool(contrast*expected_contrast>0)))
            print(corpus_name,'heldout docs',len(selected),flush=True)
    assert gauge_gate<1e-3,gauge_gate
    output=dict(source_discovery=discovery_stem,source_test_panel=PAIR_SOURCE,
        held_out_pairs=discovery['held_out_test_pairs'],
        gauge_rotation_max_abs_logit=gauge_gate,
        output_category_contrast='punctuation output probability mass minus space-word output probability mass; difference of deletion effects for punctuation versus word key variants',
        records=records)
    stem=discovery_stem+'_holdout'
    (ART/f'{stem}.json').write_text(json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--discovery-stem',default=DISCOVERY)
    a=p.parse_args();run(a.discovery_stem)

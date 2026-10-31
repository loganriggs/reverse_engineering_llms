"""Paired group and dose controls for the two-token conditional intervention."""

from __future__ import annotations

import json

import numpy as np
import torch

from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_prerope_conditional_group_ablation import KeyConditionalEdit,kl_positions
from flagship_prerope_group_ablation import candidates
from flagship_transfer import ART


SOURCE='head0_prerope_conditional_group_ablation_n16_c10_p10_pos10_ids355-523_newdocs.json'
EXCLUSION='head0_prerope_conditional_group_ablation_n8_c0_tokens.json'
DOSES=(.25,.5,1.)


def run():
    torch.set_num_threads(4)
    source=json.loads((ART/SOURCE).read_text())
    exclusion=json.loads((ART/EXCLUSION).read_text())
    excluded={corpus:{r['document_id'] for r in exclusion['records']
                      if r['corpus']==corpus} for corpus in source['corpora']}
    base,effects,details=candidates()
    config=json.loads((SNAPSHOT/'config.json').read_text())
    config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state)
    del state
    editor=KeyConditionalEdit(model)
    transported={}
    for group,effect in effects.items():
        editor.set_matrices(base,effect,128)
        transported[group]=editor.effect_matrices
    base_matrices=editor.base_matrices
    datasets={'fineweb_validation':(CORPUS,18750,20000),
              **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[]
    with torch.no_grad():
        for corpus_name in source['corpora']:
            path,begin,end=datasets[corpus_name]
            corpus=np.load(path,mmap_mode='r')
            for seed in source['sample_seeds']:
                rng=np.random.default_rng(seed+128000)
                choices=np.setdiff1d(np.arange(begin,end),
                                     list(excluded[corpus_name]))
                docs=rng.choice(choices,size=source['n_docs_per_seed'],
                                replace=False)
                expected={r['position']:r['document_id'] for r in source['records']
                          if r['corpus']==corpus_name and r['seed']==seed
                          and r['label']=='target_keys'}
                assert all(expected[i]==int(doc) for i,doc in enumerate(docs))
                for offset in range(0,len(docs),4):
                    spans=[]
                    for doc in docs[offset:offset+4]:
                        start=int(rng.integers(0,corpus.shape[1]-129+1))
                        spans.append(np.asarray(corpus[doc,start:start+129],
                                                dtype=np.int64))
                    idx=torch.from_numpy(np.stack(spans)[:,:-1].copy())
                    mask=torch.from_numpy(np.isin(idx.numpy(),
                                      source['target_token_ids']))
                    editor.key_mask=torch.zeros_like(idx,dtype=torch.bool)
                    editor.base_matrices=base_matrices
                    editor.fraction=1.
                    reference=logits(model,idx)
                    editor.key_mask=mask
                    for group,effect_matrix in transported.items():
                        editor.effect_matrices=effect_matrix
                        for dose in DOSES:
                            editor.fraction=dose
                            candidate=logits(model,idx)
                            kls=kl_positions(reference,candidate)
                            for local,doc in enumerate(docs[offset:offset+4]):
                                records.append(dict(corpus=corpus_name,
                                    seed=seed,document_id=int(doc),
                                    position=offset+local,group=group,
                                    kind=details[group]['kind'],dose=dose,
                                    n_key_tokens=int(mask[local].sum()),
                                    kl_per_position=float(kls[local].mean()),
                                    removed_head_energy=float(
                                        editor.removed_head_energy[local])))
                print(corpus_name,seed,flush=True)
    output=dict(layer=0,head=0,rank=96,
        source_artifact=SOURCE,source_document_count=source['n_docs_per_seed'],
        sample_seeds=source['sample_seeds'],doses=list(DOSES),
        target_token_ids=source['target_token_ids'],
        group_details=details,
        scope='paired full-model conditional group comparison at three deletion doses; head energy measured before output projection; same documents and spans as source',
        records=records)
    (ART/'head0_prerope_group_dose_panel.json').write_text(
        json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    run()

"""Inspect highest-effect predictions for the two-token conditional edit."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_component_labels import Decoder
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_prerope_conditional_group_ablation import KeyConditionalEdit,kl_positions
from flagship_prerope_group_ablation import candidates
from flagship_transfer import ART


SOURCE='head0_prerope_conditional_group_ablation_n16_c10_p10_pos10_ids355-523_newdocs.json'
EXCLUSION='head0_prerope_conditional_group_ablation_n8_c0_tokens.json'


def run(n_docs_per_corpus=3,n_positions_per_doc=3):
    torch.set_num_threads(4)
    decoder=Decoder()
    source=json.loads((ART/SOURCE).read_text())
    prior=json.loads((ART/EXCLUSION).read_text())
    base,effects,_=candidates()
    config=json.loads((SNAPSHOT/'config.json').read_text())
    config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state)
    del state
    editor=KeyConditionalEdit(model)
    editor.set_matrices(base,effects['identity_0'],128)
    datasets={'fineweb_validation':(CORPUS,18750,20000),
              **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[]
    with torch.no_grad():
        for corpus_name in source['corpora']:
            path,begin,end=datasets[corpus_name]
            corpus=np.load(path,mmap_mode='r')
            excluded={r['document_id'] for r in prior['records']
                      if r['corpus']==corpus_name}
            candidates_rows=[r for r in source['records']
                             if r['corpus']==corpus_name
                             and r['label']=='target_keys']
            selected=sorted(candidates_rows,
                            key=lambda r:-r['kl_per_position'])[:n_docs_per_corpus]
            by_key={(r['seed'],r['position']):r for r in selected}
            for seed in source['sample_seeds']:
                rng=np.random.default_rng(seed+128000)
                choices=np.setdiff1d(np.arange(begin,end),list(excluded))
                docs=rng.choice(choices,size=source['n_docs_per_seed'],
                                replace=False)
                for offset in range(0,len(docs),4):
                    spans=[]
                    for doc in docs[offset:offset+4]:
                        start=int(rng.integers(0,corpus.shape[1]-129+1))
                        spans.append(np.asarray(corpus[doc,start:start+129],
                                                dtype=np.int64))
                    for local,doc in enumerate(docs[offset:offset+4]):
                        position=offset+local
                        if (seed,position) not in by_key:continue
                        assert int(doc)==by_key[(seed,position)]['document_id']
                        span=spans[local]
                        idx=torch.from_numpy(span[:128][None].copy())
                        mask=torch.from_numpy(np.isin(idx.numpy(),
                                                      source['target_token_ids']))
                        editor.key_mask=torch.zeros_like(idx,dtype=torch.bool)
                        reference=logits(model,idx)
                        editor.key_mask=mask
                        edited=logits(model,idx)
                        p=F.softmax(reference.to(torch.float64),dim=-1)
                        q=F.softmax(edited.to(torch.float64),dim=-1)
                        kl=kl_positions(reference,edited)[0]
                        key_positions=np.flatnonzero(mask.numpy()[0]).tolist()
                        for t in np.argsort(kl)[-n_positions_per_doc:][::-1]:
                            t=int(t)
                            top_p=torch.topk(p[0,t],5)
                            top_q=torch.topk(q[0,t],5)
                            def predictions(top):
                                return [dict(token_id=int(i),
                                    text=decoder.decode(int(i)),
                                    probability=float(prob))
                                    for prob,i in zip(top.values,top.indices)]
                            records.append(dict(corpus=corpus_name,seed=seed,
                                document_id=int(doc),document_position=position,
                                prediction_position=t,
                                previous_target_key_positions=[j for j in key_positions
                                                               if j<=t],
                                kl_at_position=float(kl[t]),
                                context=''.join(decoder.decode(int(x))
                                                for x in span[max(0,t-24):t+1]),
                                true_next_token=decoder.decode(int(span[t+1])),
                                original_top5=predictions(top_p),
                                edited_top5=predictions(top_q)))
            print(corpus_name,flush=True)
    output=dict(source_artifact=SOURCE,
        selection='top full-model KL document windows from existing holdout; descriptive, not a new validation sample',
        n_documents_per_corpus=n_docs_per_corpus,
        n_prediction_positions_per_document=n_positions_per_doc,
        records=records)
    (ART/'head0_prerope_edited_exemplars.json').write_text(
        json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    return output


if __name__=='__main__':
    run()

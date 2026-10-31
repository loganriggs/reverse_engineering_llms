"""Native-versus-rank256 fidelity on fresh unseen-token counterfactuals."""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_shared_key_full_model import KeyDictionaryEdit
from flagship_transfer import ART


SOURCE='all_head_full_context_same_position_n24_s1331_l64_f100_unseen.json'


def kl_last(native,reduced):
    p=F.log_softmax(native[0,-1].to(torch.float64),dim=-1)
    q=F.log_softmax(reduced[0,-1].to(torch.float64),dim=-1)
    return float((p.exp()*(p-q)).sum())


def run(source=SOURCE):
    torch.set_num_threads(4)
    panel=json.loads((ART/source).read_text())
    basis=np.load(ART/'all_head_function_aware_shared_key_s642_n32768_t160.npz')['basis']
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    editor=KeyDictionaryEdit(model,panel['length'])
    corpora={'fineweb_test':CORPUS,**DATASETS}
    records=[];gate=None
    with torch.no_grad():
        for corpus_name,path in corpora.items():
            corpus=np.load(path,mmap_mode='r')
            selected=[r for r in panel['records']
                      if r['corpus']==corpus_name and r['method']=='punctuation_axis_1']
            by_doc={}
            for r in selected:by_doc.setdefault(r['document_id'],{})[r['key_category']]=r
            assert len(by_doc)==panel['n_docs_per_corpus']
            for doc,variants in by_doc.items():
                outputs={}
                for category,r in variants.items():
                    tokens=np.asarray(corpus[doc,r['start']:r['start']+panel['length']+1],
                                      dtype=np.int64).copy()
                    tokens[r['key_position']]=r['key_id']
                    idx=torch.from_numpy(tokens[:-1].copy())[None,:]
                    editor.reset();native=logits(model,idx)
                    if gate is None:
                        editor.set_shared(np.eye(9*HD,dtype=np.float32))
                        gate=float((logits(model,idx)-native).abs().max())
                        assert gate<1e-3,gate
                    editor.set_shared(basis);reduced=logits(model,idx)
                    a=native[0,-1].to(torch.float64)
                    b=reduced[0,-1].to(torch.float64)
                    outputs[category]=(a,b)
                    records.append(dict(corpus=corpus_name,document_id=int(doc),
                        key_category=category,key_id=r['key_id'],
                        last_position_kl=kl_last(native,reduced),
                        last_logit_sse=float((a-b).square().sum()),
                        last_native_logit_energy=float(a.square().sum()),
                        target_logprob_abs_difference=abs(r['native_logprob']-
                                                         r['common_baseline_logprob'])))
                native_delta=outputs['punctuation'][0]-outputs['space_word'][0]
                reduced_delta=outputs['punctuation'][1]-outputs['space_word'][1]
                records.append(dict(corpus=corpus_name,document_id=int(doc),
                    key_category='punctuation_minus_space_word',
                    response_sse=float((native_delta-reduced_delta).square().sum()),
                    native_response_energy=float(native_delta.square().sum()),
                    response_relative_mse=float((native_delta-reduced_delta).square().sum()/
                                                native_delta.square().sum())))
            print(corpus_name,'docs',len(by_doc),flush=True)
    output=dict(source=source,layer=0,rank=256,length=panel['length'],
        identity_max_abs_logit=gate,
        metric='native versus rank-256 last-position logits on held-out key-token substitutions',
        records=records)
    (ART/source.replace('.json','_reduction_fidelity.json')).write_text(
        json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default=SOURCE)
    a=p.parse_args();run(a.source)

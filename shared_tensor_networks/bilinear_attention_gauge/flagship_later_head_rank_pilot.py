"""Matched-rank, zero-intercept head cuts from weights, text, and token edits.

This is a reduction-metric pilot, not a sparsifying gauge or a claim of
canonical coordinates. All candidate cuts are orthogonal rank-r projections
on one 128-d pre-output-projection attention head at the final prompt token.
"""

import json

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from flagship_explicit_task_head_holdout import TARGETS, pairs as holdout_pairs
from flagship_explicit_task_path_patch import proposed_pairs
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT, GPTConfig
from flagship_matrix_free import CORPUS, SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART

RANKS=(4,8,16)
TRAIN_CONTEXTS=256
TEST_CONTEXTS=24
LENGTH=32
SEED=20261031


class HeadCut:
    def __init__(self, attention):
        self.native=attention.c_proj.forward
        self.n_head=attention.n_head
        self.head_dim=attention.head_dim
        self.last_input=None
        self.head=None
        self.basis=None
        self.injection=None
        self.capture_grad=False
        attention.c_proj.forward=self.forward

    def forward(self,x):
        if self.capture_grad:
            x=x.detach().requires_grad_(True)
            self.last_input=x
        else:
            self.last_input=x.detach().clone()
        if self.basis is not None:
            a=self.head*self.head_dim
            z=x[:,-1,a:a+self.head_dim]
            x=x.clone()
            reduced=(z@self.basis)@self.basis.T
            if self.injection is not None:
                reduced=reduced+self.injection
            x[:,-1,a:a+self.head_dim]=reduced
        return self.native(x)


def final_logits(model,idx,stop_at=None):
    x=model.transformer.wte(idx)
    x=F.rms_norm(x,(x.size(-1),))
    x0=x
    v1=None
    for layer,block in enumerate(model.transformer.h):
        x,v1=block(x,v1,x0)
        if stop_at is not None and layer==stop_at:
            return None
    x=F.rms_norm(x[:,-1],(x.size(-1),))
    return 30*torch.tanh(model.lm_head(x)/30)


def contexts(path,rng,n,docs,length):
    data=np.load(path,mmap_mode='r')
    d=rng.choice(np.asarray(docs),size=n,replace=False)
    starts=rng.integers(0,data.shape[1]-length+1,size=n)
    return torch.from_numpy(np.stack([
        np.asarray(data[doc,start:start+length],dtype=np.int64)
        for doc,start in zip(d,starts)]))


def top_vectors(moment):
    eig,vec=np.linalg.eigh((moment+moment.T)/2)
    return vec[:,np.argsort(eig)[::-1]].copy()


def encode_pairs(tokenizer,pairs):
    out=[]
    for family,base,edited,answer_a,answer_b in pairs:
        ids_a=tokenizer.encode(base,add_special_tokens=False).ids
        ids_b=tokenizer.encode(edited,add_special_tokens=False).ids
        answers=[tokenizer.encode(s,add_special_tokens=False).ids
                 for s in (answer_a,answer_b)]
        assert len(ids_a)==len(ids_b) and sum(a!=b for a,b in zip(ids_a,ids_b))==1
        assert len(answers[0])==len(answers[1])==1
        out.append((family,ids_a,ids_b,answers[0][0],answers[1][0]))
    return out


def capture_head_states(model,probes,ids):
    final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:],stop_at=max(probes))
    return {layer:p.last_input[0,-1].reshape(p.n_head,p.head_dim).clone()
            for layer,p in probes.items()}


def metrics(reference,candidate):
    a=F.log_softmax(reference.double(),dim=-1)
    b=F.log_softmax(candidate.double(),dim=-1)
    return ((a.exp()*(a-b)).sum(-1).cpu().numpy(),
            (reference.argmax(-1)!=candidate.argmax(-1)).cpu().numpy())


def run(fit_only=False):
    torch.set_num_threads(4)
    rng=np.random.default_rng(SEED)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probes={layer:HeadCut(model.transformer.h[layer].attn)
            for layer in sorted(set(layer for layer,_ in TARGETS.values()))}
    assert {p.head_dim for p in probes.values()}=={128}
    testsets={'fineweb_test':contexts(CORPUS,rng,TEST_CONTEXTS,range(17500,20000),LENGTH)}
    for name,path in DATASETS.items():
        testsets[name]=contexts(path,rng,TEST_CONTEXTS,range(0,1200),LENGTH)
    fit_text=contexts(CORPUS,rng,TRAIN_CONTEXTS,range(0,16000),LENGTH)
    train_pairs=encode_pairs(tokenizer,proposed_pairs())
    test_pairs=encode_pairs(tokenizer,holdout_pairs())
    print('Contexts and pairs',len(fit_text),len(train_pairs),len(test_pairs),flush=True)
    corpus_vectors={family:[] for family in TARGETS}
    task_deltas={family:[] for family in TARGETS}
    with torch.no_grad():
        for batch in fit_text.split(8):
            final_logits(model,batch,stop_at=max(probes))
            for family,(layer,head) in TARGETS.items():
                p=probes[layer];a=head*p.head_dim
                corpus_vectors[family].append(p.last_input[:,-1,a:a+p.head_dim].numpy())
        for _,before,after,_,_ in train_pairs:
            a=capture_head_states(model,probes,before)
            b=capture_head_states(model,probes,after)
            for family,(layer,head) in TARGETS.items():
                task_deltas[family].append((b[layer][head]-a[layer][head]).numpy())
    bases={}
    eig_info={}
    for family,(layer,head) in TARGETS.items():
        p=probes[layer]
        z=np.concatenate(corpus_vectors[family],axis=0).astype(np.float64)
        d=np.stack(task_deltas[family]).astype(np.float64)
        w=p.native.__self__.weight.detach().numpy()[:,head*p.head_dim:(head+1)*p.head_dim].astype(np.float64)
        m_weight=w.T@w
        m_corpus=z.T@z/len(z)
        m_task=d.T@d/len(d)
        def unit_trace(m):return m/max(float(np.trace(m)),1e-30)
        m_mixed=unit_trace(m_corpus)+unit_trace(m_task)
        random=np.linalg.qr(rng.standard_normal((p.head_dim,p.head_dim)))[0]
        bases[family]={'weight':top_vectors(m_weight),'corpus':top_vectors(m_corpus),
                       'task_delta':top_vectors(m_task),'mixed':top_vectors(m_mixed),
                       'random':random}
        eig_info[family]=dict(weight_trace=float(np.trace(m_weight)),
                              corpus_trace=float(np.trace(m_corpus)),
                              task_delta_trace=float(np.trace(m_task)),
                              task_delta_count=len(d))
    np.savez(ART/'later_head_rank_pilot_bases.npz',
             **{f'{family}__{method}':vectors for family,options in bases.items()
                for method,vectors in options.items()})
    if fit_only:
        print('Saved bases only',flush=True)
        return
    # Keep independent native references fixed for every comparison.
    task_refs=[];corpus_refs={}
    with torch.no_grad():
        for family,before,after,answer_a,answer_b in test_pairs:
            logits_a=final_logits(model,torch.tensor(before)[None,:])[0].detach().clone()
            logits_b=final_logits(model,torch.tensor(after)[None,:])[0].detach().clone()
            task_refs.append((family,before,after,answer_a,answer_b,logits_a,logits_b))
        for corpus_name,x in testsets.items():
            corpus_refs[corpus_name]=torch.cat([final_logits(model,b) for b in x.split(8)]).detach()
    records=[]
    with torch.no_grad():
        for family,(layer,head) in TARGETS.items():
            probe=probes[layer]
            probe.head=head
            for method,vectors in bases[family].items():
                for rank in RANKS:
                    probe.basis=torch.tensor(vectors[:,:rank].copy(),dtype=torch.float32)
                    task_rows=[]
                    for test_family,before,after,answer_a,answer_b,ref_a,ref_b in task_refs:
                        if test_family!=family:continue
                        cut_a=final_logits(model,torch.tensor(before)[None,:])[0]
                        cut_b=final_logits(model,torch.tensor(after)[None,:])[0]
                        native_a=float(ref_a[answer_a]-ref_a[answer_b])
                        native_b=float(ref_b[answer_a]-ref_b[answer_b])
                        margin_a=float(cut_a[answer_a]-cut_a[answer_b])
                        margin_b=float(cut_b[answer_a]-cut_b[answer_b])
                        task_rows.append(dict(native_edit=native_b-native_a,
                                              cut_edit=margin_b-margin_a,
                                              both_correct=margin_a>0 and margin_b<0,
                                              kl_base=float(metrics(ref_a[None,:],cut_a[None,:])[0][0]),
                                              kl_edited=float(metrics(ref_b[None,:],cut_b[None,:])[0][0])))
                    native=np.array([x['native_edit'] for x in task_rows])
                    cut=np.array([x['cut_edit'] for x in task_rows])
                    row=dict(family=family,layer=layer,head=head,method=method,rank=rank,
                             task_pairs=len(task_rows),
                             task_both_correct=sum(x['both_correct'] for x in task_rows),
                             task_edit_effect_mae=float(np.mean(np.abs(cut-native))),
                             task_edit_effect_relative_mae=float(np.mean(np.abs(cut-native))/np.mean(np.abs(native))),
                             task_kl_mean=float(np.mean([y for x in task_rows for y in (x['kl_base'],x['kl_edited'])])))
                    for corpus_name,x in testsets.items():
                        out=torch.cat([final_logits(model,b) for b in x.split(8)])
                        kl,flips=metrics(corpus_refs[corpus_name],out)
                        row[f'{corpus_name}_kl_mean']=float(np.mean(kl))
                        row[f'{corpus_name}_top1_flips']=int(np.sum(flips))
                    records.append(row)
                    print(family,method,rank,'task relative',round(row['task_edit_effect_relative_mae'],3),
                          'fineweb kl',round(row['fineweb_test_kl_mean'],4),flush=True)
            probe.basis=None
    result=dict(description='Orthogonal zero-intercept rank cuts at one final-position later-layer attention head; separate weight, natural, task-delta, mixed, random bases',
                seed=SEED,ranks=RANKS,train_contexts=TRAIN_CONTEXTS,test_contexts_per_corpus=TEST_CONTEXTS,
                context_length=LENGTH,train_pairs=len(train_pairs),holdout_pairs=len(test_pairs),
                fit_fineweb_docs=[0,16000],test_fineweb_docs=[17500,20000],
                eig_info=eig_info,records=records,
                caveat='Task-delta basis sees only discovery prompt edits, not holdout edits. Head-output reduction is not an exact gauge or a canonical functional decomposition.')
    path=ART/'later_head_rank_pilot.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',path,flush=True)
    return result


if __name__=='__main__':
    import sys
    run(fit_only='--fit-only' in sys.argv)

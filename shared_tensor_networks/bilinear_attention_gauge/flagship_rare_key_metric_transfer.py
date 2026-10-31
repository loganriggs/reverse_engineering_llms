"""Full-model rare-key transfer of fixed pre-RoPE rank-96 metric cuts.

Select natural rare/common key occurrences by FineWeb *training* frequency.
On held-out FineWeb, WikiText-103, and TinyStories documents, replace only
that key's layer-0/head-0 QK1 contribution by uniform, mixed, or corpus
rank-96 cuts. Evaluate paired native-to-cut suffix KL through all 18 layers.
"""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matched_rope_full_model import PositionAwareBondEdit
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART


PRIORS=('uniform','mixed','corpus')
LENGTH=64
TARGET_POSITION=16


class KeyMaskedPreBondEdit(PositionAwareBondEdit):
    def __init__(self,model):
        super().__init__(model)
        self.key_mask=None
        self.removed_output_energy=None

    def forward(self,q,k,v,q2,k2):
        if self.pre_cut is None:
            return super().forward(q,k,v,q2,k2)
        assert self.key_mask is not None
        native=self.native(q,k,v,q2,k2)
        d=q.shape[-1]
        matrices=self.pre_matrices.to(device=q.device,dtype=q.dtype)
        qx=torch.einsum('btd,tde->bte',q[:,:,0,:],matrices)
        key=k[:,:,0,:]
        score1=(q[:,:,0,:]@key.transpose(-1,-2))/d
        score1_cut=(qx@key.transpose(-1,-2))/d
        score2=(q2[:,:,0,:]@k2[:,:,0,:].transpose(-1,-2))/d
        score_delta=(score1_cut-score1)*self.key_mask[:,None,:]
        head_delta=(score_delta*score2).tril()@v[:,:,0,:]
        output_weight=self.attn.c_proj.weight[:,:d]
        self.removed_output_energy=(head_delta@output_weight.T).square().sum(
            dim=(1,2)).cpu().numpy()
        result=native.clone()
        result[:,0]=((score1+score_delta)*score2).tril()@v[:,:,0,:]
        return result


def frequency_counts():
    train=np.load(CORPUS,mmap_mode='r')
    return np.bincount(np.asarray(train[:15000]).ravel(),minlength=50304)


def select_windows(corpus_name,path,begin,end,counts,n,seed):
    corpus=np.load(path,mmap_mode='r')
    rng=np.random.default_rng(seed)
    chosen=[];used=set()
    docs=rng.permutation(np.arange(begin,end))
    for stratum in ('rare','common'):
        for doc in docs:
            if len([x for x in chosen if x['stratum']==stratum])>=n:break
            if int(doc) in used:continue
            tokens=np.asarray(corpus[doc],dtype=np.int64)
            valid=np.arange(TARGET_POSITION,len(tokens)-(LENGTH-TARGET_POSITION))
            local=counts[tokens[valid]]
            if stratum=='rare':
                eligible=valid[(local>=1)&(local<10)]
            else:
                eligible=valid[local>=1000]
            if not len(eligible):continue
            position=int(rng.choice(eligible))
            start=position-TARGET_POSITION
            window=tokens[start:start+LENGTH+1]
            assert len(window)==LENGTH+1
            target_id=int(window[TARGET_POSITION])
            chosen.append(dict(corpus=corpus_name,stratum=stratum,
                document_id=int(doc),document_start=start,
                key_position=TARGET_POSITION,key_token_id=target_id,
                train_token_frequency=int(counts[target_id]),
                ids=window.tolist()))
            used.add(int(doc))
    assert all(sum(x['stratum']==s for x in chosen)==n for s in ('rare','common'))
    assert len({x['document_id'] for x in chosen})==2*n
    return chosen


def metrics(reference,candidate,target,key_position):
    p=F.log_softmax(reference.to(torch.float64),dim=-1)
    q=F.log_softmax(candidate.to(torch.float64),dim=-1)
    kl=(p.exp()*(p-q)).sum(-1)
    ce_change=-(q-p).gather(-1,target[...,None]).squeeze(-1)
    return dict(suffix_kl=kl[:,key_position:].mean(1).numpy(),
                whole_kl=kl.mean(1).numpy(),
                prefix_max_abs_kl=kl[:,:key_position].abs().max(1).values.numpy(),
                suffix_ce_change=ce_change[:,key_position:].mean(1).numpy(),
                max_abs_logit=(candidate-reference).abs().flatten(1).max(1).values.numpy())


def run(n=24,seed=20264100,batch_size=4,strata=('rare','common')):
    torch.set_num_threads(4)
    counts=frequency_counts()
    corpora={'fineweb_test':(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    cases=[]
    for index,(name,(path,begin,end)) in enumerate(corpora.items()):
        cases.extend(c for c in select_windows(name,path,begin,end,counts,n,seed+index)
                     if c['stratum'] in strata)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    editor=KeyMaskedPreBondEdit(model)
    arrays=np.load(ART/'head0_rope_metric_factorial_seed41.npz')
    cuts={name:arrays[f'{name}__pre'] for name in PRIORS}
    records=[];gates={}
    with torch.no_grad():
        for corpus_name in corpora:
            for stratum in strata:
                indices=[i for i,c in enumerate(cases)
                         if c['corpus']==corpus_name and c['stratum']==stratum]
                for start in range(0,len(indices),batch_size):
                    group=indices[start:start+batch_size]
                    tokens=torch.tensor([cases[i]['ids'] for i in group])
                    x,target=tokens[:,:-1],tokens[:,1:]
                    mask=torch.zeros_like(x,dtype=torch.bool)
                    mask[:,TARGET_POSITION]=True
                    editor.set_cut('pre',np.eye(HD),LENGTH)
                    editor.pre_cut=None
                    editor.pre_matrices=None
                    editor.key_mask=None
                    reference=logits(model,x)
                    editor.set_cut('pre',np.eye(HD),LENGTH)
                    editor.key_mask=mask
                    identity=logits(model,x)
                    gate=float((identity-reference).abs().max())
                    assert gate<1e-4,gate
                    gates['identity']=max(gates.get('identity',0.),gate)
                    for name,cut in cuts.items():
                        editor.set_cut('pre',cut,LENGTH)
                        editor.key_mask=torch.zeros_like(mask)
                        null=logits(model,x)
                        null_gate=float((null-reference).abs().max())
                        assert null_gate<1e-4,null_gate
                        gates['null_mask']=max(gates.get('null_mask',0.),null_gate)
                        editor.key_mask=mask
                        candidate=logits(model,x)
                        result=metrics(reference,candidate,target,TARGET_POSITION)
                        assert float(np.max(result['prefix_max_abs_kl']))<1e-8
                        for j,i in enumerate(group):
                            records.append(dict(case_index=i,prior=name,
                                removed_post_c_proj_energy=float(editor.removed_output_energy[j]),
                                **{key:float(value[j]) for key,value in result.items()}))
                print(corpus_name,stratum,'done',flush=True)
    result=dict(description='First-layer head-0 QK1 pre-RoPE rank-96 uniform/mixed/corpus cuts from unchanged seed-41 fit. Key-masked intervention at one naturally occurring rare (FineWeb train count 1–9) or common (count >=1000) token per disjoint held-out document; all other keys native. Three corpora, full 18-layer propagation, per-case suffix KL, CE change, and post-c_proj edit energy. Token-frequency selection uses only FineWeb train docs 0–14999. No fitting or prior selection on these cases.',
        n_per_stratum_per_corpus=n,seed=seed,strata=list(strata),length=LENGTH,
        key_position=TARGET_POSITION,fit_cut_seed=41,
        frequency_corpus_documents=[0,15000],
        fineweb_test_documents=[17500,20000],
        frequency_bands=dict(rare=[1,10],common=[1000,None]),
        identity_gates=gates,cases=cases,records=records)
    suffix='' if set(strata)=={'rare','common'} else '_'+ '-'.join(strata)
    (ART/f'head0_pre_rank96_rare_key_transfer_n{n}_s{seed}{suffix}.json').write_text(
        json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=24)
    p.add_argument('--seed',type=int,default=20264100)
    p.add_argument('--strata',choices=('rare','common'),nargs='+',
                   default=['rare','common'])
    a=p.parse_args()
    run(a.n,a.seed,strata=tuple(a.strata))

"""Full-model group deletion conditioned on connective or matched key tokens.

The selected rank-96 pre-RoPE cut and the group are fixed before this test.
Matched token sets come from the training-corpus frequency distribution.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_component_labels import Decoder,check_decoder
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_prerope_cut_transfer import row_rotation
from flagship_prerope_group_ablation import candidates
from flagship_prerope_group_frequency_null import matched_sets
from flagship_transfer import ART


class KeyConditionalEdit:
    def __init__(self,model):
        self.attn=model.transformer.h[0].attn
        self.native=self.attn.squared_attention
        self.attn.squared_attention=self.forward
        self.base_matrices=None
        self.effect_matrices=None
        self.key_mask=None
        self.fraction=1.
        self.removed_head_energy=None

    def set_matrices(self,base,effect,length):
        def transport(matrix):
            return torch.from_numpy(np.stack([
                np.linalg.solve(row_rotation(p),matrix@row_rotation(p))
                for p in range(length)
            ])).to(torch.float32)
        self.base_matrices=transport(base)
        self.effect_matrices=transport(effect)

    def forward(self,q,k,v,q2,k2):
        native=self.native(q,k,v,q2,k2)
        if self.base_matrices is None:
            return native
        assert self.key_mask is not None
        d=q.shape[-1]
        base=self.base_matrices.to(device=q.device,dtype=q.dtype)
        effect=self.effect_matrices.to(device=q.device,dtype=q.dtype)
        query=q[:,:,0,:]
        key=k[:,:,0,:]
        score1=torch.einsum('btd,tde->bte',query,base)@key.transpose(-1,-2)/d
        removed=torch.einsum('btd,tde->bte',query,effect)@key.transpose(-1,-2)/d
        score2=q2[:,:,0,:]@k2[:,:,0,:].transpose(-1,-2)/d
        key_mask=self.key_mask.to(device=q.device).unsqueeze(1)
        delta=self.fraction*(((removed*score2)*key_mask).tril()@v[:,:,0,:])
        self.removed_head_energy=delta.square().sum(dim=(1,2)).cpu().numpy()
        output_weight=self.attn.c_proj.weight[:,:d]
        self.removed_head_output_energy=(delta@output_weight.T).square().sum(
            dim=(1,2)).cpu().numpy()
        result=native.clone()
        result[:,0]=(score1*score2).tril()@v[:,:,0,:]-delta
        return result


def kl_positions(reference,candidate):
    p=F.log_softmax(reference.to(torch.float64),dim=-1)
    q=F.log_softmax(candidate.to(torch.float64),dim=-1)
    return (p.exp()*(p-q)).sum(-1).numpy()


def nearby_position_controls(target_mask,n_controls,rng,window=8):
    """Same key count per document, replacing each key by a nearby key."""
    result=[]
    for _ in range(n_controls):
        picked=np.zeros_like(target_mask,dtype=bool)
        for row in range(target_mask.shape[0]):
            targets=np.flatnonzero(target_mask[row])
            for j in targets:
                nearby=np.arange(max(0,j-window),
                                 min(target_mask.shape[1],j+window+1))
                available=nearby[~target_mask[row,nearby]&~picked[row,nearby]]
                if len(available)==0:
                    available=np.flatnonzero(~target_mask[row]&~picked[row])
                assert len(available)>0
                picked[row,int(rng.choice(available))]=True
        assert np.array_equal(picked.sum(axis=1),target_mask.sum(axis=1))
        result.append(picked)
    return result


def run(n_docs=4,sample_seeds=(23,29),n_null=10,batch_size=4,
        pool_size=100,n_position_null=0,per_token=False,
        target_token_ids=None,exclude_document_records=None,
        corpora=('fineweb_validation','wikitext103','tinystories')):
    torch.set_num_threads(4)
    decoder=Decoder();check_decoder(decoder)
    fineweb=np.load(CORPUS,mmap_mode='r')
    target_ids,controls,_,_=matched_sets(decoder,fineweb,n_null,
        pool_size=pool_size,target_ids=target_token_ids)
    excluded={}
    if exclude_document_records is not None:
        prior=json.loads((ART/exclude_document_records).read_text())
        for row in prior['records']:
            excluded.setdefault(row['corpus'],set()).add(row['document_id'])
    base,effects,_=candidates()
    effect=effects['identity_0']
    config=json.loads((SNAPSHOT/'config.json').read_text())
    config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state)
    del state
    editor=KeyConditionalEdit(model)
    editor.set_matrices(base,effect,128)
    datasets={'fineweb_validation':(CORPUS,18750,20000),
              **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[];identity_gate=None;native_to_cut_max_abs_logit=None
    all_key_equivalence_gate=None
    with torch.no_grad():
        for corpus_name in corpora:
            path,begin,end=datasets[corpus_name]
            corpus=np.load(path,mmap_mode='r')
            for seed in sample_seeds:
                rng=np.random.default_rng(seed+128000)
                choices=np.setdiff1d(np.arange(begin,end),
                                     list(excluded.get(corpus_name,set())))
                docs=rng.choice(choices,size=n_docs,replace=False)
                for offset in range(0,n_docs,batch_size):
                    spans=[]
                    for doc in docs[offset:offset+batch_size]:
                        start=int(rng.integers(0,corpus.shape[1]-129+1))
                        spans.append(np.asarray(corpus[doc,start:start+129],
                                                dtype=np.int64))
                    idx=torch.from_numpy(np.stack(spans)[:,:-1].copy())
                    editor.key_mask=torch.zeros_like(idx,dtype=torch.bool)
                    reference=logits(model,idx)
                    if identity_gate is None:
                        repeated=logits(model,idx)
                        identity_gate=float((repeated-reference).abs().max())
                        assert identity_gate<1e-6
                        editor.base_matrices=None
                        native=logits(model,idx)
                        editor.set_matrices(base,effect,128)
                        native_to_cut_max_abs_logit=float(
                            (native-reference).abs().max())
                        editor.key_mask=torch.ones_like(idx,dtype=torch.bool)
                        all_key_deleted=logits(model,idx)
                        original_base=editor.base_matrices
                        original_effect=editor.effect_matrices
                        editor.set_matrices(base-effect,np.zeros_like(effect),128)
                        editor.key_mask=torch.zeros_like(idx,dtype=torch.bool)
                        cut_deleted=logits(model,idx)
                        all_key_equivalence_gate=float(
                            (all_key_deleted-cut_deleted).abs().max())
                        assert all_key_equivalence_gate<1e-3
                        editor.base_matrices=original_base
                        editor.effect_matrices=original_effect
                    target_mask=np.isin(idx.numpy(),target_ids)
                    position_rng=np.random.default_rng(seed+515000+offset)
                    position_masks=nearby_position_controls(
                        target_mask,n_position_null,position_rng)
                    target_label=('connectives' if target_token_ids is None
                                  else 'target_keys')
                    masks=[(target_label,target_mask)]+[
                        (f'matched_{i}',np.isin(idx.numpy(),x))
                        for i,x in enumerate(controls)]+[
                        (f'nearby_position_{i}',x)
                        for i,x in enumerate(position_masks)]
                    if per_token:
                        present=np.intersect1d(np.unique(idx.numpy()),target_ids)
                        masks.extend((f'target_token_{int(token)}',
                                      idx.numpy()==token) for token in present)
                    for label,mask_array in masks:
                        mask=torch.from_numpy(mask_array)
                        editor.key_mask=mask
                        candidate=logits(model,idx)
                        kls=kl_positions(reference,candidate)
                        after=mask.cumsum(dim=-1).numpy()>0
                        for local,doc in enumerate(docs[offset:offset+batch_size]):
                            records.append(dict(corpus=corpus_name,seed=seed,
                                document_id=int(doc),position=offset+local,
                                label=label,n_key_tokens=int(mask[local].sum()),
                                n_after_positions=int(after[local].sum()),
                                kl_per_position=float(kls[local].mean()),
                                kl_after_key=(float(kls[local,after[local]].mean())
                                              if after[local].any() else None),
                                removed_head_energy=float(editor.removed_head_energy[local])))
                print(corpus_name,seed,flush=True)
    output=dict(layer=0,head=0,rank=96,n_docs_per_seed=n_docs,
        sample_seeds=list(sample_seeds),n_frequency_matched_null=n_null,
        matched_frequency_pool_size=pool_size,
        n_nearby_position_null=n_position_null,
        nearby_position_window=8,
        per_token=per_token,
        target_token_text={str(int(x)):decoder.decode(x)
                           for x in target_ids},
        corpora=list(corpora),target_token_ids=target_ids.tolist(),
        matched_null_token_ids=[x.tolist() for x in controls],
        matched_null_seed=910,
        excluded_document_artifact=exclude_document_records,
        excluded_document_counts={k:len(v) for k,v in excluded.items()},
        zero_mask_repeat_max_abs_logit=identity_gate,
        all_key_vs_full_cut_max_abs_logit=all_key_equivalence_gate,
        native_to_cut_max_abs_logit=native_to_cut_max_abs_logit,
        fineweb_train_frequency_documents=[0,15000],
        fineweb_validation_documents=[18750,20000],
        control_rule='same coarse token form and one of the nearest FineWeb training frequencies per token ID',
        scope='conditional deletion of the same exact-gauge group from the fixed mixed pre-RoPE cut; full 18-layer model; no semantic task outcome',
        records=records)
    suffix=f'n{n_docs}_c{n_null}'
    if pool_size!=100:suffix+=f'_p{pool_size}'
    if n_position_null:suffix+=f'_pos{n_position_null}'
    if per_token:suffix+='_tokens'
    if target_token_ids is not None:
        suffix+='_ids'+'-'.join(str(int(x)) for x in target_ids)
    if exclude_document_records is not None:suffix+='_newdocs'
    (ART/f'head0_prerope_conditional_group_ablation_{suffix}.json').write_text(
        json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n-docs',type=int,default=4)
    p.add_argument('--n-null',type=int,default=10)
    p.add_argument('--pool-size',type=int,default=100)
    p.add_argument('--position-nulls',type=int,default=0)
    p.add_argument('--per-token',action='store_true')
    p.add_argument('--target-token-ids',type=int,nargs='+',default=None)
    p.add_argument('--exclude-document-records',default=None)
    p.add_argument('--seeds',type=int,nargs='+',default=[23,29])
    p.add_argument('--corpora',nargs='+',default=['fineweb_validation','wikitext103','tinystories'])
    a=p.parse_args()
    run(a.n_docs,tuple(a.seeds),a.n_null,pool_size=a.pool_size,
        n_position_null=a.position_nulls,per_token=a.per_token,
        target_token_ids=a.target_token_ids,
        exclude_document_records=a.exclude_document_records,
        corpora=tuple(a.corpora))

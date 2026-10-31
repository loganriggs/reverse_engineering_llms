"""Fresh-document finite deletions of full-context QK1 sensitivity axes.

Selection uses only the sensitivity-gauge training documents. Test windows
pair punctuation and space-word keys at nearby positions before reading logits.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from flagship_component_labels import CAT_INDEX,CATEGORIES,Decoder
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_shared_key_dictionary import project_all_heads
from flagship_shared_key_full_model import KeyDictionaryEdit
from flagship_transfer import ART


STEM='all_head_full_context_sensitivity_gauge_train64_test24_l64_s1031'


class OneKeyAxisDeletion(KeyDictionaryEdit):
    def __init__(self,model,length):
        super().__init__(model,length)
        self.axis=None
        self.key_position=None
        self.fraction=1.
        self.last_attention=None

    def forward(self,q,k,v,q2,k2):
        if self.axis is None:
            result=super().forward(q,k,v,q2,k2)
            self.last_attention=result.detach()
            return result
        basis=self.shared.to(device=k.device,dtype=k.dtype)
        inverse=self.inv.to(device=k.device,dtype=k.dtype)
        rotation=self.rot.to(device=k.device,dtype=k.dtype)
        pre=torch.einsum('bthd,tde->bthe',k,inverse)
        flat=pre.reshape(-1,pre.shape[-2]*pre.shape[-1])
        coordinates=(flat@basis).reshape(pre.shape[0],pre.shape[1],-1)
        reduced=(coordinates@basis.T).reshape_as(pre)
        correction=(self.fraction*coordinates[:,self.key_position,self.axis]
                    .reshape(-1,1,1)*basis[:,self.axis].reshape(1,9,HD))
        reduced[:,self.key_position]-=correction
        post=torch.einsum('bthd,tde->bthe',reduced,rotation)
        result=self.native(q,post,v,q2,k2)
        self.last_attention=result.detach()
        return result


def stable_axis_cosines(rotation,independent):
    scores=[]
    for name in ('full_context_random_start_0','full_context_random_start_1'):
        similarity=np.abs(rotation.T@independent[name])
        i,j=linear_sum_assignment(-similarity)
        aligned=np.empty(rotation.shape[1]);aligned[i]=similarity[i,j]
        scores.append(aligned)
    return np.minimum(*scores)


def axis_profiles(state,basis,train,rotations,decoder):
    keys=train['train_fineweb_keys']
    gradients=train['train_fineweb_vectors'].astype(np.float64)
    unique,inverse=np.unique(keys,return_inverse=True)
    key_table=project_all_heads(state,unique,('c_k',))['c_k'].reshape(-1,9*HD)
    categories=np.array([decoder.category(token) for token in unique])[inverse]
    base=np.bincount(categories,minlength=len(CATEGORIES))/len(categories)
    profiles={}
    for name,rotation in rotations.items():
        axes=basis@rotation
        effect=(gradients@rotation)*(key_table[inverse]@axes)
        energy=effect**2
        fraction=energy/np.maximum(energy.sum(1,keepdims=True),1e-30)
        total=energy.sum(0);total/=total.sum()
        enriched=[]
        for axis in range(256):
            by=np.bincount(categories,weights=fraction[:,axis],
                           minlength=len(CATEGORIES))
            by/=max(by.sum(),1e-30)
            category=int(np.argmax(np.where(base>=.02,by/np.maximum(base,1e-30),-1)))
            enriched.append(dict(axis=axis,energy_share=float(total[axis]),
                proposed_category=CATEGORIES[category],
                row_normalized_enrichment=float(by[category]/base[category])))
        profiles[name]=enriched
    return profiles


def select_methods(state,basis,train,decoder):
    saved=np.load(ART/f'{STEM}_independent_starts.npz')
    old=np.load(ART/'all_head_shared_key_gauge_priors_function_n2048_p2048_i120_d600.npz')
    rotations={'full_context':saved['full_context_varimax'],
               'random':old['random_0']}
    profiles=axis_profiles(state,basis,train,rotations,decoder)
    stable=stable_axis_cosines(rotations['full_context'],saved)
    candidates=[row for row in profiles['full_context']
        if row['proposed_category']=='punctuation'
        and row['energy_share']>=.003
        and stable[row['axis']]>=.9]
    candidates.sort(key=lambda row:row['row_normalized_enrichment'],reverse=True)
    assert len(candidates)>=2,candidates
    chosen=candidates[:2]
    target_energy=chosen[0]['energy_share']
    other=[row for row in profiles['full_context']
           if row['proposed_category']!='punctuation'
           and row['energy_share']>=.003 and stable[row['axis']]>=.9]
    same=min(other,key=lambda row:abs(np.log(row['energy_share']/target_energy)))
    random=min(profiles['random'],
               key=lambda row:abs(np.log(row['energy_share']/target_energy)))
    methods={f'punctuation_axis_{i+1}':('full_context',row['axis'])
             for i,row in enumerate(chosen)}
    methods['same_gauge_other_label']=('full_context',same['axis'])
    methods['random_gauge_matched_energy']=('random',random['axis'])
    detail={name:dict(gauge=gauge,**profiles[gauge][axis],
                      independent_start_min_abs_cosine=float(stable[axis])
                      if gauge=='full_context' else None)
            for name,(gauge,axis) in methods.items()}
    return rotations,methods,detail


def fresh_windows(corpus,begin,end,excluded,decoder,n_docs,seed,length):
    rng=np.random.default_rng(seed)
    choices=np.setdiff1d(np.arange(begin,end),np.asarray(sorted(excluded),dtype=int))
    rng.shuffle(choices)
    windows=[];examined=0
    for doc in choices:
        examined+=1
        start=int(rng.integers(0,corpus.shape[1]-length-1))
        tokens=np.asarray(corpus[doc,start:start+length+1],dtype=np.int64)
        cats=np.asarray([decoder.category(token) for token in tokens[:-2]])
        positions=np.arange(length-1)
        eligible=positions>=max(0,length-32)
        punctuation=positions[eligible&(cats==CAT_INDEX['punctuation'])]
        word=positions[eligible&(cats==CAT_INDEX['space_word'])]
        if not len(punctuation) or not len(word):continue
        gap=np.abs(punctuation[:,None]-word[None,:])
        i,j=np.unravel_index(np.argmin(gap),gap.shape)
        windows.append(dict(document_id=int(doc),start=start,
            tokens=tokens,positions={'punctuation':int(punctuation[i]),
                                     'space_word':int(word[j])},
            position_gap=int(gap[i,j])))
        if len(windows)>=n_docs:break
    assert len(windows)==n_docs,(len(windows),n_docs,examined)
    return windows,examined


def logprob_last(output,target):
    return float(F.log_softmax(output[0,-1].float(),dim=-1)[target])


def run(n_docs=8,seed=1131,length=64,fraction=1.):
    torch.set_num_threads(4)
    assert length==64 and 0<fraction<=1
    decoder=Decoder()
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
    exclusions={name:set(prior['corpus_splits'][f'test_{name}'
                    if name!='fineweb_test' else 'test_fineweb'])
                for name in corpora}
    records=[];coverage={};gate=None
    gauge_gate=0.;zero_gate=0.;gauge_logprob_gate=0.;zero_logprob_gate=0.
    with torch.no_grad():
        for corpus_index,(corpus_name,(path,begin,end)) in enumerate(corpora.items()):
            corpus=np.load(path,mmap_mode='r')
            windows,examined=fresh_windows(corpus,begin,end,
                exclusions[corpus_name],decoder,n_docs,seed+100000*corpus_index,length)
            coverage[corpus_name]=dict(examined_documents=examined,
                selected_documents=len(windows),mean_position_gap=float(np.mean(
                    [entry['position_gap'] for entry in windows])))
            for entry in windows:
                idx=torch.from_numpy(entry['tokens'][:-1].copy())[None,:]
                target=int(entry['tokens'][-1])
                editor.axis=None;editor.reset()
                native=logits(model,idx)
                editor.set_shared(basis)
                baseline=logits(model,idx)
                if gate is None:
                    editor.set_shared(np.eye(9*HD,dtype=np.float32))
                    gate=float((logits(model,idx)-native).abs().max())
                    assert gate<1e-3,gate
                base_lp=logprob_last(baseline,target)
                native_lp=logprob_last(native,target)
                for name,(gauge,axis) in methods.items():
                    editor.set_shared((basis@rotations[gauge]).astype(np.float32))
                    editor.axis=None
                    gauge_baseline=logits(model,idx)
                    gauge_baseline_head=editor.last_attention[:, -1].clone()
                    gauge_base_lp=logprob_last(gauge_baseline,target)
                    gauge_gate=max(gauge_gate,float((gauge_baseline-baseline).abs().max()))
                    gauge_logprob_gate=max(gauge_logprob_gate,abs(gauge_base_lp-base_lp))
                    editor.axis=axis;editor.fraction=fraction
                    if len(records)<len(methods)*2:
                        editor.key_position=length-2
                        editor.fraction=0.
                        zero=logits(model,idx)
                        zero_gate=max(zero_gate,float((zero-gauge_baseline).abs().max()))
                        zero_logprob_gate=max(zero_logprob_gate,
                            abs(logprob_last(zero,target)-gauge_base_lp))
                        editor.fraction=fraction
                    for category,position in entry['positions'].items():
                        editor.key_position=position
                        edited=logits(model,idx)
                        changed=logprob_last(edited,target)-gauge_base_lp
                        dose=float((editor.last_attention[:, -1]-gauge_baseline_head).square().sum())
                        records.append(dict(corpus=corpus_name,
                            document_id=entry['document_id'],start=entry['start'],
                            target_id=target,method=name,key_category=category,
                            key_position=position,key_id=int(entry['tokens'][position]),
                            position_gap=entry['position_gap'],
                            baseline_logprob=gauge_base_lp,
                            common_baseline_logprob=base_lp,native_logprob=native_lp,
                            logprob_change=changed,logprob_change_sq=changed**2,
                            head_output_change_sq=dose))
                    editor.axis=None
            print(corpus_name,'docs',len(windows),flush=True)
    assert gauge_gate<1e-3 and zero_gate<1e-3,(gauge_gate,zero_gate)
    output=dict(layer=0,heads=list(range(9)),rank=256,length=length,
        n_docs_per_corpus=n_docs,seed=seed,deletion_fraction=fraction,
        selection='train-only: two highest row-normalized punctuation-enrichment axes with >=0.3% predicted deletion-effect energy and >=0.9 cross-fit cosine; controls closest in predicted training effect energy',
        edit='remove one coordinate of one pre-RoPE QK1 key in fixed rank-256 model',
        readout='last-position next-token log probability after all 18 layers',
        identity_max_abs_logit=gate,
        gauge_rotation_max_abs_logit=gauge_gate,
        zero_deletion_max_abs_logit=zero_gate,
        gauge_rotation_max_abs_target_logprob=gauge_logprob_gate,
        zero_deletion_max_abs_target_logprob=zero_logprob_gate,
        detail=detail,coverage=coverage,records=records,
        caveat='conditional punctuation/space-word key comparison is not a token-edit counterfactual or physical-neuron identification')
    stem=f'all_head_full_context_finite_axis_n{n_docs}_s{seed}_l{length}_f{round(fraction*100)}'
    (ART/f'{stem}.json').write_text(json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n-docs',type=int,default=8)
    p.add_argument('--seed',type=int,default=1131)
    p.add_argument('--length',type=int,default=64)
    p.add_argument('--fraction',type=float,default=1.)
    a=p.parse_args()
    run(a.n_docs,a.seed,a.length,a.fraction)

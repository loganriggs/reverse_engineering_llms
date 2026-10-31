"""Full-model input-edit test of candidate shared-key component labels.

The component is selected using FineWeb validation labels only. For a held-out
window with a space-prefixed query token, replace an earlier space-prefixed
key by punctuation or by another space-prefixed word. Compare how much a
one-component deletion changes each input-edit response in final logits.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_component_labels import CAT_INDEX,Decoder,check_decoder
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_function_aware_shared_key import make_panels
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_shared_key_full_model import KeyDictionaryEdit
from flagship_shared_key_label_transfer import component_energy
from flagship_transfer import ART


class CaptureEdit(KeyDictionaryEdit):
    def forward(self,q,k,v,q2,k2):
        result=super().forward(q,k,v,q2,k2)
        self.last_attention=result.detach().clone()
        return result


def candidates(state,basis,rotations):
    report=json.loads((ART/'all_head_shared_key_label_transfer_n32768_clip99.json').read_text())
    detail={};selected=[]
    for gauge in ('response_0','direct_l1_native_0'):
        eligible=[row for row in report['records']
                  if row['law']=='validation_fineweb' and row['gauge']==gauge
                  and row['proposed_label']=='space_word -> punctuation'
                  and row['selection_rank']<10]
        row=max(eligible,key=lambda r:r['log2_enrichment'])
        selected.append((gauge,row['component']))
        detail[gauge]=dict(component=row['component'],
                           selection_rank=row['selection_rank'],
                           validation_log2_enrichment=row['log2_enrichment'])
    panels,_,_=make_panels(state,128,4096,128)
    panel=panels['validation_fineweb']
    random_u=basis@rotations['random_0']
    random_energy=component_energy(panel,random_u).sum(0)
    taken=set();methods={}
    for gauge,component in selected:
        u=basis@rotations[gauge]
        energy=float(component_energy(panel,u)[:,component].sum())
        order=np.argsort(np.abs(np.log(np.maximum(random_energy,1e-30)/
                                       max(energy,1e-30))))
        control=next(int(c) for c in order if int(c) not in taken)
        taken.add(control)
        name=f'{gauge}_c{component}'
        control_energy=float(random_energy[control])
        dose=float(np.sqrt(control_energy/energy))
        assert 0<dose<=1,(name,dose)
        methods[name+'_full']=np.delete(u,component,axis=1)
        partial=u.copy()
        partial[:,component]*=np.sqrt(1-dose)
        methods[name+'_matched']=partial
        detail[name+'_full']=dict(kind='candidate_full',gauge=gauge,
            component=component,selection_pair_energy=energy,dose=1.)
        detail[name+'_matched']=dict(kind='candidate_matched',gauge=gauge,
            component=component,selection_pair_energy=energy,dose=dose,
            matched_selection_pair_energy=dose*dose*energy)
        control_name=f'random_matched_to_{gauge}_c{control}'
        methods[control_name]=np.delete(random_u,control,axis=1)
        detail[control_name]=dict(kind='matched_random',gauge='random_0',
                   component=control,selection_pair_energy=control_energy,
                   matched_to=name+'_matched',dose=1.,
                   full_candidate_to_control_energy_ratio=float(energy/control_energy))
    return methods,detail


def windows(corpus,begin,end,seed,n_docs,length,decoder,excluded,
            punctuation_ids,control_word_ids):
    rng=np.random.default_rng(seed+length*1000)
    choices=np.setdiff1d(np.arange(begin,end),list(excluded))
    rng.shuffle(choices)
    output=[];skipped=0
    for doc in choices:
        for _ in range(4):
            start=int(rng.integers(0,corpus.shape[1]-length))
            tokens=np.asarray(corpus[doc,start:start+length],dtype=np.int64)
            cat=np.array([decoder.category(t) for t in tokens])
            candidates=[]
            for key in range(16,min(96,length-12)):
                if cat[key]!=CAT_INDEX['space_word']:continue
                queries=np.flatnonzero(cat[key+1:key+13]==CAT_INDEX['space_word'])
                if len(queries):candidates.append((key,key+1+int(queries[0])))
            if not candidates:continue
            key,query=candidates[int(rng.integers(len(candidates)))]
            assert key<query and cat[query]==CAT_INDEX['space_word']
            punctuation=punctuation_ids[len(output)%len(punctuation_ids)]
            word=control_word_ids[len(output)%len(control_word_ids)]
            if word==tokens[key]:
                word=control_word_ids[(len(output)+1)%len(control_word_ids)]
            assert word!=tokens[key]
            variants=np.stack((tokens,tokens.copy(),tokens.copy()))
            variants[1,key]=punctuation
            variants[2,key]=word
            output.append(dict(document_id=int(doc),start=start,key=key,
                query=query,original_key_id=int(tokens[key]),
                punctuation_id=punctuation,control_word_id=word,
                variants=variants))
            break
        else:skipped+=1
        if len(output)==n_docs:break
    assert len(output)==n_docs,(len(output),n_docs)
    return output,skipped


def norm2(x):
    return float(np.sum(np.asarray(x,dtype=np.float64)**2))


def run(n_docs=8,seeds=(91,97),length=128,batch_size=2,
        punctuation_ids=(13,11),control_word_ids=(262,290),
        exclude_causal_artifacts=()):
    torch.set_num_threads(4)
    decoder=Decoder();check_decoder(decoder)
    assert len(punctuation_ids)>=2 and len(control_word_ids)>=2
    assert all(decoder.category(t)==CAT_INDEX['punctuation']
               for t in punctuation_ids)
    assert all(decoder.category(t)==CAT_INDEX['space_word']
               for t in control_word_ids)
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    basis=np.load(ART/'all_head_function_aware_shared_key_s642_n32768_t160.npz')[
          'basis'].astype(np.float64)
    rotation_file='all_head_shared_key_gauge_priors_function_n2048_p2048_i120_d600.npz'
    rotations=np.load(ART/rotation_file)
    methods,detail=candidates(state,basis,rotations)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval();model.load_state_dict(state)
    del state
    editor=CaptureEdit(model,length)
    excluded={}
    for path in ART.glob('all_head_shared_key_full_model_n16*.json'):
        prior=json.loads(path.read_text())
        for row in prior['records']:
            excluded.setdefault(row['corpus'],set()).add(row['document_id'])
    for filename in exclude_causal_artifacts:
        prior=json.loads((ART/filename).read_text())
        for row in prior['records']:
            excluded.setdefault(row['corpus'],set()).add(row['document_id'])
    corpora={'fineweb_test':(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[];skipped={};identity_gate=None
    with torch.no_grad():
        for corpus_name,(path,begin,end) in corpora.items():
            corpus=np.load(path,mmap_mode='r')
            used=excluded.get(corpus_name,set()).copy()
            for seed in seeds:
                entries,missed=windows(corpus,begin,end,seed,n_docs,length,
                    decoder,used,punctuation_ids,control_word_ids)
                skipped[f'{corpus_name}_{seed}']=missed
                used.update(entry['document_id'] for entry in entries)
                for offset in range(0,len(entries),batch_size):
                    chunk=entries[offset:offset+batch_size]
                    stack=np.concatenate([entry['variants'] for entry in chunk],axis=0)
                    idx=torch.from_numpy(stack.copy())
                    if identity_gate is None:
                        editor.reset();native=logits(model,idx)
                        editor.set_shared(np.eye(9*HD,dtype=np.float32))
                        identity_gate=float((logits(model,idx)-native).abs().max())
                        assert identity_gate<1e-3,identity_gate
                    editor.set_shared(basis)
                    reference=logits(model,idx).numpy()
                    reference_head=editor.last_attention.numpy()
                    for name,cut in methods.items():
                        editor.set_shared(cut)
                        candidate=logits(model,idx).numpy()
                        candidate_head=editor.last_attention.numpy()
                        for local,entry in enumerate(chunk):
                            b=3*local;i=entry['query']
                            delta_p=reference[b+1,i]-reference[b,i]
                            delta_w=reference[b+2,i]-reference[b,i]
                            med_p=delta_p-(candidate[b+1,i]-candidate[b,i])
                            med_w=delta_w-(candidate[b+2,i]-candidate[b,i])
                            h_p=((reference_head[b+1,:,i]-reference_head[b,:,i])-
                                 (candidate_head[b+1,:,i]-candidate_head[b,:,i]))
                            h_w=((reference_head[b+2,:,i]-reference_head[b,:,i])-
                                 (candidate_head[b+2,:,i]-candidate_head[b,:,i]))
                            p=F.log_softmax(torch.from_numpy(reference[b,i]).to(torch.float64),dim=-1)
                            q=F.log_softmax(torch.from_numpy(candidate[b,i]).to(torch.float64),dim=-1)
                            kl=float(torch.sum(p.exp()*(p-q)))
                            records.append(dict(corpus=corpus_name,seed=seed,
                                document_id=entry['document_id'],start=entry['start'],
                                key_position=entry['key'],query_position=i,
                                original_key_id=entry['original_key_id'],
                                punctuation_id=entry['punctuation_id'],
                                control_word_id=entry['control_word_id'],
                                method=name,base_query_kl=kl,
                                base_head_deletion_energy=norm2(reference_head[b,:,i]-candidate_head[b,:,i]),
                                punctuation_logit_response_energy=norm2(delta_p),
                                word_logit_response_energy=norm2(delta_w),
                                punctuation_mediated_logit_energy=norm2(med_p),
                                word_mediated_logit_energy=norm2(med_w),
                                punctuation_mediated_head_energy=norm2(h_p),
                                word_mediated_head_energy=norm2(h_w)))
                print(corpus_name,seed,flush=True)
    result=dict(layer=0,rank=256,length=length,n_docs_per_seed=n_docs,
                seeds=list(seeds),batch_size=batch_size,
                candidate_selection='top-10 validation-energy components whose validation-only proposed label is space_word -> punctuation; maximize validation enrichment',
                control_selection='random-gauge axis with closest validation pair-output energy, without replacement; candidate partial dose matches its validation energy',
                input_edit='replace an earlier space-prefixed word key by one of the recorded punctuation tokens or by a different recorded space-prefixed word; keep later space-prefixed query fixed',
                punctuation_ids=list(punctuation_ids),
                control_word_ids=list(control_word_ids),
                exclude_causal_artifacts=list(exclude_causal_artifacts),
                baseline='same rank-256 reduced model; each method removes one coordinate',
                identity_projection_max_abs_logit=identity_gate,
                source_basis='all_head_function_aware_shared_key_s642_n32768_t160.npz',
                source_rotations=rotation_file,detail=detail,
                excluded_full_model_document_counts={k:len(v) for k,v in excluded.items()},
                skipped_windows=skipped,records=records,
                scope='full-model input-edit mediation probe, not a proof of semantic or physical circuit localization')
    suffix=f'n{n_docs}_s'+ '-'.join(map(str,seeds))
    if tuple(punctuation_ids)!=(13,11) or tuple(control_word_ids)!=(262,290):
        suffix+='_punct'+'-'.join(map(str,punctuation_ids))
        suffix+='_word'+'-'.join(map(str,control_word_ids))
    (ART/f'all_head_shared_key_counterfactual_causality_{suffix}.json').write_text(
        json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n-docs',type=int,default=8)
    p.add_argument('--seeds',type=int,nargs='+',default=[91,97])
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--punctuation-ids',type=int,nargs='+',default=[13,11])
    p.add_argument('--control-word-ids',type=int,nargs='+',default=[262,290])
    p.add_argument('--exclude-causal-artifacts',nargs='+',default=[])
    a=p.parse_args()
    run(a.n_docs,tuple(a.seeds),batch_size=a.batch_size,
        punctuation_ids=tuple(a.punctuation_ids),
        control_word_ids=tuple(a.control_word_ids),
        exclude_causal_artifacts=tuple(a.exclude_causal_artifacts))

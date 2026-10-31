"""Post-fit held-out token-form labels for shared nine-head key gauges."""

import json

import numpy as np
import torch

from flagship_component_labels import CATEGORIES,Decoder,check_decoder,label_name
from flagship_function_aware_shared_key import make_panels
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_shared_key_dictionary import HEADS,WIDTH,sampled_pairs
from flagship_transfer import ART


def component_energy(panel,u,chunk=1024):
    q=panel['q'].numpy().astype(np.float64)
    k=panel['k'].numpy().astype(np.float64)
    s2=panel['s2'].numpy().astype(np.float64)
    gram=panel['gram'].numpy().astype(np.float64)
    energy=np.empty((len(k),u.shape[1]),dtype=np.float64)
    u3=u.reshape(HEADS,HD,-1)
    for start in range(0,len(k),chunk):
        stop=min(start+chunk,len(k))
        coef=k[start:stop]@u
        query=np.einsum('nhd,hdr->nhr',q[start:stop],u3,optimize=True)
        scores=(query*coef[:,None,:]*s2[start:stop,:,None]/HD).transpose(0,2,1)
        energy[start:stop]=np.einsum('nrh,nhj,nrj->nr',
                          scores,gram[start:stop],scores,optimize=True)
    assert np.min(energy)>-1e-4
    return np.maximum(energy,0)


def fractions(energy,labels):
    ncat=len(CATEGORIES)**2
    by=np.stack([np.bincount(labels,weights=energy[:,c],minlength=ncat)
                 for c in range(energy.shape[1])],axis=0)
    return by/np.maximum(by.sum(1,keepdims=True),1e-30)


def run(n_pairs=4096,top_components=20,winsorize=False):
    torch.set_num_threads(4)
    decoder=Decoder();check_decoder(decoder)
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    vocab=state['transformer.wte.weight'].shape[0]
    offset=30000
    panels,gate,_=make_panels(state,128,n_pairs,n_pairs,
                               test_seed_offset=offset)
    del state
    laws={
        'validation_fineweb':('corpus',CORPUS,15000,17500,2902),
        'test_fineweb':('corpus',CORPUS,17500,20000,3902+offset),
        **{f'test_{name}':('corpus',path,0,1200,3903+i+offset)
           for i,(name,path) in enumerate(DATASETS.items())},
    }
    labels={};base={};coverage={};sample_weights={}
    for law,(distribution,path,begin,end,seed) in laws.items():
        qid,kid,_,_=sampled_pairs(distribution,path,begin,end,n_pairs,
                                  seed,vocab)
        qcat=np.array([decoder.category(x) for x in qid],dtype=np.int64)
        kcat=np.array([decoder.category(x) for x in kid],dtype=np.int64)
        labels[law]=qcat*len(CATEGORIES)+kcat
        energy=panels[law]['ref'].numpy().astype(np.float64)
        cap=float(np.quantile(energy,.99)) if winsorize else float('inf')
        sample_weights[law]=np.minimum(1.,cap/np.maximum(energy,1e-30))
        adjusted=energy*sample_weights[law]
        by=np.bincount(labels[law],weights=adjusted,
                       minlength=len(CATEGORIES)**2)
        base[law]=by/max(by.sum(),1e-30)
        coverage[law]=dict(pairs=len(labels[law]),
                           response_energy_ess=float(adjusted.sum()**2/
                           np.maximum(np.sum(adjusted**2),1e-30)),
                           energy_cap=cap if winsorize else None)
    source='all_head_function_aware_shared_key_s642_n32768_t160.npz'
    basis=np.load(ART/source)['basis'].astype(np.float64)
    rotations_name='all_head_shared_key_gauge_priors_function_n2048_p2048_i120_d600.npz'
    rotations=np.load(ART/rotations_name)
    methods=('native','random_0','weight_0','response_0',
             'direct_l1_native_0','direct_l1_native_1',
             'direct_l1_random_0')
    records=[];examples={}
    for name in methods:
        u=basis@rotations[name]
        train=component_energy(panels['validation_fineweb'],u)*\
              sample_weights['validation_fineweb'][:,None]
        train_fraction=fractions(train,labels['validation_fineweb'])
        eligible=base['validation_fineweb']>=.02
        order=np.argsort(train.sum(0))[::-1][:top_components]
        chosen={}
        for c in order:
            enrichment=np.where(eligible,
                train_fraction[c]/np.maximum(base['validation_fineweb'],1e-30),
                -1)
            chosen[int(c)]=int(np.argmax(enrichment))
        for law in laws:
            component=fractions(component_energy(panels[law],u)*
                                sample_weights[law][:,None],labels[law])
            for rank,c in enumerate(order):
                category=chosen[int(c)]
                frac=float(component[c,category])
                baseline=float(base[law][category])
                records.append(dict(gauge=name,law=law,component=int(c),
                    selection_rank=rank,proposed_label=label_name(category),
                    component_fraction=frac,base_fraction=baseline,
                    log2_enrichment=float(np.log2(max(frac,1e-30)/
                                                   max(baseline,1e-30)))))
        if name in ('native','direct_l1_native_0'):
            sample=sampled_pairs('corpus',CORPUS,15000,17500,n_pairs,
                                 2902,vocab)
            qid,kid=sample[:2]
            for rank,c in enumerate(order[:8]):
                chosen_category=chosen[int(c)]
                mask=labels['validation_fineweb']==chosen_category
                indices=np.argsort(np.where(mask,train[:,c],-1))[::-1][:5]
                examples[f'{name}_{int(c)}']=[dict(query=decoder.decode(qid[i]),
                      key=decoder.decode(kid[i]),component_energy=float(train[i,c]))
                      for i in indices]
        print('evaluated',name,flush=True)
    output=dict(rank=256,heads=list(range(HEADS)),n_pairs_per_law=n_pairs,
                tail_handling='per-law 99th-percentile native-energy clipping' if winsorize else 'raw response energy',
                top_components=top_components,
                source_basis=source,source_rotations=rotations_name,
                selection_fineweb_documents=[15000,17500],
                test_fineweb_documents=[17500,20000],
                labels='coarse query-token-form -> key-token-form categories, used only after gauge fitting',
                categories=list(CATEGORIES),
                selection='top components by selection pair-output energy; category with maximum enrichment among categories carrying at least 2% native output energy',
                formula_gate=gate,coverage=coverage,
                records=records,examples=examples,
                scope='post-fit label transfer; token-form categories are limited proxies for semantic computation')
    suffix='_clip99' if winsorize else ''
    path=ART/f'all_head_shared_key_label_transfer_n{n_pairs}{suffix}.json'
    path.write_text(json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--n-pairs',type=int,default=4096)
    p.add_argument('--top-components',type=int,default=20)
    p.add_argument('--winsorize',action='store_true')
    a=p.parse_args()
    run(a.n_pairs,a.top_components,a.winsorize)

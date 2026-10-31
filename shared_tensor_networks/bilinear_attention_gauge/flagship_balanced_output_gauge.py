"""Balanced next-token-category sensitivity panel for a fixed 546M QK1 cut.

Collect full-context gradients on disjoint FineWeb document splits and two
external corpora. Fit an exact rotation using only four target-token categories;
hold two entire categories out of gauge fitting. Category is a task stratum,
not proof that the model has a separate category circuit.
"""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_component_labels import CATEGORIES, Decoder
from flagship_full_context_sensitivity_gauge import SensitivityEdit, effective_coordinates, normalize_rows
from flagship_full_model_eval import GPT, GPTConfig, logits
from flagship_matrix_free import CORPUS, HD, SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_shared_key_gauge_priors import varimax
from flagship_transfer import ART


LENGTH = 64
CATEGORY_NAMES = ('digits', 'punctuation', 'space_word',
                  'capitalized_word', 'word_fragment', 'mixed')
FIT_NAMES = ('digits', 'punctuation', 'space_word', 'word_fragment')
HELD_NAMES = ('capitalized_word', 'mixed')
BASE_BASIS = 'all_head_function_aware_shared_key_s642_n32768_t160.npz'
BASE_ROTATIONS = 'all_head_shared_key_gauge_priors_function_n2048_p2048_i120_d600.npz'


def select_spans(path, begin, end, n_per_category, categories, seed):
    corpus = np.load(path, mmap_mode='r')
    rng = np.random.default_rng(seed)
    category_ids = np.array([CATEGORIES.index(name) for name in categories])
    decoder = Decoder()
    category_map = np.array([decoder.category(i) for i in range(50304)], dtype=np.int8)
    shuffled = rng.permutation(np.arange(begin, end))
    used = set()
    selected = []
    for name, category in zip(categories, category_ids):
        count = 0
        for doc in shuffled:
            if int(doc) in used:
                continue
            possible = np.where(category_map[np.asarray(corpus[doc, LENGTH:])] == category)[0]
            if len(possible) == 0:
                continue
            target_position = LENGTH + int(rng.choice(possible))
            tokens = np.asarray(corpus[doc, target_position-LENGTH:target_position+1],
                                dtype=np.int64).copy()
            selected.append({'doc_id': int(doc), 'target_position': target_position,
                             'category': name, 'tokens': tokens})
            used.add(int(doc)); count += 1
            if count == n_per_category:
                break
        if count != n_per_category:
            raise RuntimeError(f'{name}: found {count}/{n_per_category} docs in {path}')
    rng.shuffle(selected)
    return selected


def panel(model, editor, basis, selections):
    vectors=[];categories=[];keys=[];doc_ids=[];target_ids=[];target_positions=[]
    target_logprobs=[];top1=[];gate=None
    basis_t = torch.from_numpy(basis.astype(np.float32))
    for item in selections:
        tokens=item['tokens']
        idx=torch.from_numpy(tokens[:-1].copy())[None,:]
        target=int(tokens[-1])
        output=logits(model,idx)[0,-1].float()
        logprob=F.log_softmax(output,dim=-1)[target]
        grad=torch.autograd.grad(logprob,editor.last_pre)[0][0]
        projected=(grad.reshape(LENGTH,-1)@basis_t).detach().numpy()
        assert np.isfinite(projected).all()
        if gate is None:
            axis=int(np.argmax(np.abs(projected[-2])))
            editor.injection=(LENGTH-2,basis_t[:,axis],.1)
            plus=F.log_softmax(logits(model,idx)[0,-1].float(),dim=-1)[target]
            editor.injection=(LENGTH-2,basis_t[:,axis],-.1)
            minus=F.log_softmax(logits(model,idx)[0,-1].float(),dim=-1)[target]
            editor.injection=None
            finite=float(((plus-minus)/.2).detach())
            analytic=float(projected[-2,axis])
            gate={'axis':axis,'analytic':analytic,'central_difference':finite,
                  'absolute_error':abs(analytic-finite)}
            assert gate['absolute_error'] < max(2e-4,.05*abs(analytic)),gate
        vectors.append(projected)
        categories.extend([CATEGORIES.index(item['category'])]*LENGTH)
        keys.extend(map(int,tokens[:-1]))
        doc_ids.extend([item['doc_id']]*LENGTH)
        target_ids.append(target)
        target_positions.append(item['target_position'])
        target_logprobs.append(float(logprob.detach()))
        top1.append(int(output.argmax()) == target)
    return {'vectors':np.concatenate(vectors).astype(np.float32),
            'categories':np.asarray(categories,dtype=np.int8),
            'keys':np.asarray(keys,dtype=np.int32),
            'document_ids':np.asarray(doc_ids,dtype=np.int32),
            'target_ids':np.asarray(target_ids,dtype=np.int32),
            'target_positions':np.asarray(target_positions,dtype=np.int32),
            'target_logprobs':np.asarray(target_logprobs,dtype=np.float32),
            'top1':np.asarray(top1,dtype=bool),
            'finite_difference_gate':gate}


def collect(train_per_category=2, validation_per_category=2,
            test_per_category=2, external_per_category=1, seed=9071):
    torch.set_num_threads(4)
    basis=np.load(ART/BASE_BASIS)['basis']
    assert basis.shape==(9*HD,256)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    editor=SensitivityEdit(model,LENGTH,basis)
    settings={
        'train_fineweb':(CORPUS,0,15000,train_per_category,CATEGORY_NAMES,seed),
        'validation_fineweb':(CORPUS,15000,17500,validation_per_category,CATEGORY_NAMES,seed+1),
        'test_fineweb':(CORPUS,17500,20000,test_per_category,CATEGORY_NAMES,seed+2),
        'test_wikitext103':(DATASETS['wikitext103'],0,1200,external_per_category,
                            CATEGORY_NAMES,seed+3),
        'test_tinystories':(DATASETS['tinystories'],0,1200,external_per_category,
                            tuple(x for x in CATEGORY_NAMES if x!='digits'),seed+4),
    }
    panels={};metadata={}
    for name,(path,begin,end,n,categories,panel_seed) in settings.items():
        selected=select_spans(path,begin,end,n,categories,panel_seed)
        data=panel(model,editor,basis,selected)
        panels[name]=data
        metadata[name]=[{'doc_id':int(item['doc_id']),
                         'target_position':int(item['target_position']),
                         'target_id':int(item['tokens'][-1]),
                         'category':item['category']} for item in selected]
        print(name,len(selected),'docs',flush=True)
    stem=f'all_head_balanced_output_gauge_train{train_per_category}_val{validation_per_category}_test{test_per_category}_ext{external_per_category}_s{seed}'
    np.savez_compressed(ART/f'{stem}.npz',
        **{f'{name}_{field}':value for name,data in panels.items()
           for field,value in data.items() if isinstance(value,np.ndarray)})
    result={'stem':stem,'rank':256,'length':LENGTH,'seed':seed,
            'basis':BASE_BASIS,'categories':CATEGORY_NAMES,
            'fit_categories':FIT_NAMES,'held_categories':HELD_NAMES,
            'splits':metadata,
            'identity_max_abs_head_output':editor.head_equivalence_max_abs,
            'finite_difference_gates':{name:data['finite_difference_gate']
                                       for name,data in panels.items()},
            'scope':'full 18-layer observed-next-token logprob gradient through fixed rank-256 QK1 key projection'}
    (ART/f'{stem}.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def fit(stem):
    path=ART/f'{stem}.json'
    result=json.loads(path.read_text())
    arrays=np.load(ART/f'{stem}.npz')
    train=arrays['train_fineweb_vectors'].astype(np.float64)
    labels=arrays['train_fineweb_categories']
    fit_ids=[CATEGORIES.index(x) for x in FIT_NAMES]
    used=np.isin(labels,fit_ids)
    assert used.sum()>0
    fit_normalized,floor=normalize_rows(train[used])
    pooled_normalized,pooled_floor=normalize_rows(train)
    held_normalized,held_floor=normalize_rows(train[~used])
    rotation,trace=varimax(fit_normalized,np.eye(256),max_iter=240)
    pooled,pooled_trace=varimax(pooled_normalized,np.eye(256),max_iter=240)
    held,held_trace=varimax(held_normalized,np.eye(256),max_iter=240)
    controls=np.load(ART/BASE_ROTATIONS)
    choices={'native':np.eye(256),
             'weight_0':controls['weight_0'],
             'response_0':controls['response_0'],
             'balanced_fit_categories':rotation,
             'pooled_all_categories_oracle':pooled,
             'held_categories_oracle':held}
    scores={}
    for name,rot in choices.items():
        assert np.max(np.abs(rot.T@rot-np.eye(256)))<2e-3,name
        scores[name]={}
        for panel_name in result['splits']:
            vec=arrays[f'{panel_name}_vectors'].astype(np.float64)
            cats=arrays[f'{panel_name}_categories']
            per_category={}
            for category in CATEGORY_NAMES:
                subset=cats==CATEGORIES.index(category)
                if not subset.any():continue
                coord=vec[subset]@rot
                per_category[category]={'n_positions':int(subset.sum()),
                    'effective_coordinates':effective_coordinates(coord),
                    'mean_gradient_norm':float(np.mean(np.linalg.norm(coord,axis=1)))}
            scores[name][panel_name]=per_category
    result['gauge_fit']={'row_normalization_floor':floor,
                         'oracle_row_normalization_floors':{
                             'pooled_all_categories':pooled_floor,
                             'held_categories':held_floor},
                         'fit_positions':int(used.sum()),
                         'rotation_orthogonality_max_abs':{
                             name:float(np.max(np.abs(rot.T@rot-np.eye(256))))
                             for name,rot in choices.items()},
                         'varimax_traces':{'fit_categories':trace,
                                           'pooled_all_categories_oracle':pooled_trace,
                                           'held_categories_oracle':held_trace},
                         'scores':scores,
                         'caveat':'target-token categories are output strata, not separately verified tasks; gradient sparsity is not finite causal fidelity'}
    path.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ART/f'{stem}_rotations.npz',
                        **{name:rot for name,rot in choices.items()})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-per-category',type=int,default=2)
    parser.add_argument('--validation-per-category',type=int,default=2)
    parser.add_argument('--test-per-category',type=int,default=2)
    parser.add_argument('--external-per-category',type=int,default=1)
    parser.add_argument('--seed',type=int,default=9071)
    parser.add_argument('--fit-stem')
    args=parser.parse_args()
    if args.fit_stem:fit(args.fit_stem)
    else:collect(args.train_per_category,args.validation_per_category,
                 args.test_per_category,args.external_per_category,args.seed)

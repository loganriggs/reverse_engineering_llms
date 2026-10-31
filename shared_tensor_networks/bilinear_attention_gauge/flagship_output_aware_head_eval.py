"""Test output-aware and larger-data PCA cuts on paired text and edits."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from flagship_bracket_group_competent_transfer import margin
from flagship_explicit_task_head_holdout import pairs as lexical_pairs
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_format_shift import proposed_pairs as format_pairs
from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_rank_pilot import HeadCut, contexts, final_logits
from flagship_later_head_rank4_kl_scale import PANELS, SEED
from flagship_matrix_free import CORPUS, SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART


METHODS = ('weight','corpus_256','corpus_2048','output_aware_2048')


def head_state(model, probe, ids, layer, head):
    final_logits(model, torch.tensor(ids,dtype=torch.long)[None,:],stop_at=layer)
    start=head*probe.head_dim
    return probe.last_input[0,-1,start:start+probe.head_dim].clone()


def run(basis_filename='output_aware_head_fit_bases.npz', methods=METHODS,
        output_filename='output_aware_head_eval.json', check_old=True):
    torch.set_num_threads(4)
    rng=np.random.default_rng(SEED)
    config=json.loads((SNAPSHOT/'config.json').read_text())
    config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state)
    del state
    model.requires_grad_(False)
    probes={layer:HeadCut(model.transformer.h[layer].attn)
            for layer in sorted({layer for layer,_ in TARGETS.values()})}
    bases=np.load(ART/basis_filename)
    old=json.loads((ART/'later_head_rank4_kl_scale.json').read_text()) if check_old else None
    records=[]
    with torch.no_grad():
        for length,count in PANELS:
            for corpus_name,path,docs in [('fineweb_test',CORPUS,range(17500,20000))]+[
                (name,path,range(0,1200)) for name,path in DATASETS.items()]:
                x=contexts(path,rng,count,docs,length)
                native=torch.cat([final_logits(model,batch) for batch in x.split(8)])
                native_log=F.log_softmax(native.double(),dim=-1)
                native_probs=native_log.exp()
                for target,(layer,head) in TARGETS.items():
                    probe=probes[layer];probe.head=head
                    for method in methods:
                        basis=bases[f'{target}__{method}']
                        assert np.linalg.norm(basis.T@basis-np.eye(4))<1e-10
                        probe.basis=torch.tensor(basis.copy(),dtype=torch.float32)
                        candidate=torch.cat([final_logits(model,batch) for batch in x.split(8)])
                        probe.basis=None
                        kl=(native_probs*(native_log-F.log_softmax(candidate.double(),dim=-1))).sum(-1).numpy()
                        flips=(native.argmax(-1)!=candidate.argmax(-1)).numpy()
                        record=dict(target=target,corpus=corpus_name,length=length,count=count,
                            method=method,kl=kl.tolist(),mean_kl=float(kl.mean()),
                            top1_flips=flips.astype(int).tolist(),flip_count=int(flips.sum()))
                        if check_old and method in ('weight','corpus_256'):
                            previous=next(r for r in old['records'] if
                                r['target']==target and r['corpus']==corpus_name and
                                r['length']==length and
                                r['method']==('weight' if method=='weight' else 'corpus'))
                            assert np.max(np.abs(kl-np.asarray(previous['kl'])))<1e-7
                        records.append(record)
                        print('KL',target,corpus_name,length,method,
                              round(record['mean_kl'],7),flush=True)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    bracket=json.loads((ART/'competent_bracket_identities.json').read_text())
    test_cases={}
    test_cases['bracket_closure']=[dict(panel='identity',before=x['base_ids'],
        after=x['edited_ids'],a=tokenizer.encode(')',add_special_tokens=False).ids[0],
        b=tokenizer.encode(']',add_special_tokens=False).ids[0])
        for x in bracket['rows'] if x['panel']=='validation']
    agreement=[]
    for source,panel in ((lexical_pairs(),'lexical'),(format_pairs(),'format')):
        for family,base,edited,answer_a,answer_b in source:
            if family!='subject_verb_agreement':continue
            agreement.append(dict(panel=panel,
                before=tokenizer.encode(base,add_special_tokens=False).ids,
                after=tokenizer.encode(edited,add_special_tokens=False).ids,
                a=tokenizer.encode(answer_a,add_special_tokens=False).ids[0],
                b=tokenizer.encode(answer_b,add_special_tokens=False).ids[0]))
    test_cases['subject_verb_agreement']=agreement
    assert len(test_cases['bracket_closure'])==21 and len(agreement)==20
    edit_rows=[]
    with torch.no_grad():
        for target,cases in test_cases.items():
            layer,head=TARGETS[target]
            probe=probes[layer];probe.head=head
            for case in cases:
                before,after,a,b=case['before'],case['after'],case['a'],case['b']
                delta=(head_state(model,probe,after,layer,head)-
                       head_state(model,probe,before,layer,head))[None,:]
                native_base=margin(model,before,a,b)
                native_edited=margin(model,after,a,b)
                assert native_base>0 and native_edited<0
                probe.basis=torch.eye(probe.head_dim)
                probe.injection=delta
                native_path=margin(model,before,a,b)-native_base
                probe.injection=None
                effects={};task_margins={}
                for method in methods:
                    basis=torch.tensor(bases[f'{target}__{method}'].copy(),dtype=torch.float32)
                    probe.basis=basis
                    baseline=margin(model,before,a,b)
                    cut_edited=margin(model,after,a,b)
                    probe.injection=(delta@basis)@basis.T
                    effects[method]=margin(model,before,a,b)-baseline
                    probe.injection=None
                    task_margins[method]=dict(base=baseline,edited=cut_edited)
                probe.basis=None
                edit_rows.append(dict(target=target,panel=case['panel'],
                    before=before,after=after,native_head_path=native_path,
                    effects=effects,task_margins=task_margins))
                if len(edit_rows)%10==0:print('edit paths',len(edit_rows),flush=True)
    edit_summary=[]
    for target,cases in test_cases.items():
        for panel in sorted({x['panel'] for x in cases}|{'all'}):
            subset=[x for x in edit_rows if x['target']==target and
                    (panel=='all' or x['panel']==panel)]
            norm=np.mean([abs(x['native_head_path']) for x in subset])
            for method in methods:
                edit_summary.append(dict(target=target,panel=panel,method=method,
                    count=len(subset),mean_abs_native_head_path=float(norm),
                    relative_mae_vs_native=float(np.mean([
                        abs(x['effects'][method]-x['native_head_path'])
                        for x in subset])/norm),
                    both_forced_choice_correct=sum(x['task_margins'][method]['base']>0
                        and x['task_margins'][method]['edited']<0 for x in subset)))
    result=dict(description='Rank-four later-head cuts on the same 576-context-per-head natural KL panel as rank4_kl_scale, plus competent bracket/agreement finite head paths',
        fit_source=basis_filename,natural_test_seed=SEED,
        methods=methods,kl_records=records,edit_rows=edit_rows,edit_summary=edit_summary)
    (ART/output_filename).write_text(json.dumps(result,indent=2)+'\n')
    for item in edit_summary:
        if item['panel']=='all':print('path',item,flush=True)


if __name__=='__main__':run()

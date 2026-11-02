"""Audit v8 bilinear teachers on common fresh oracle and causal-edit panels."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from eval_v8_rule_holdout_readout import scope_panel, source_panel
from eval_v8_section_attention_paths import predict
from surface_mixture_v8 import SurfaceMixtureV8
from train_surface_mixture import evaluate as joint_evaluate, make_eval


def endpoint_predictions(model,pairs,pad_id,device,batch_size):
    prefixes=[]
    positions={}
    before=[]
    after=[]
    for pair in pairs:
        indices=[]
        for field in ('original','edited'):
            key=tuple(pair[field])
            if key not in positions:
                positions[key]=len(prefixes)
                prefixes.append(list(key))
            indices.append(positions[key])
        before.append(indices[0])
        after.append(indices[1])
    probabilities=predict(model,prefixes,None,None,None,pad_id,device,
                          batch_size)
    # Right padding may save computation, but must not change valid answers.
    with torch.no_grad():
        for i in range(min(3,len(prefixes))):
            raw=model(torch.tensor(prefixes[i],device=device)[None])[0]
            error=(raw[0,-1].softmax(-1).cpu()-probabilities[i]).abs().max()
            if error>1e-5:
                raise AssertionError(f'Padding changed answer {i}: {error}')
    return probabilities,torch.tensor(before),torch.tensor(after)


def edit_scores(pairs,probabilities,before,after,vocab):
    rows={}
    for kind in sorted({pair['kind'] for pair in pairs}):
        selected=[i for i,pair in enumerate(pairs) if pair['kind']==kind]
        correct=[]
        ce=[]
        effect_error=[]
        effect_size=[]
        oracle_size=[]
        for i in selected:
            pair=pairs[i]
            p0=probabilities[before[i]]
            p1=probabilities[after[i]]
            q0=pair['q_before']
            q1=pair['q_after']
            assert len(q0)==len(q1)==1
            assert next(iter(q0.values()))==next(iter(q1.values()))==1.
            t0=vocab[next(iter(q0))]
            t1=vocab[next(iter(q1))]
            correct.append((float(p0.argmax()==t0)+float(p1.argmax()==t1))/2)
            ce.append(float((-p0[t0].clamp_min(1e-30).log()-
                             p1[t1].clamp_min(1e-30).log())/2))
            oracle=torch.zeros_like(p0)
            oracle[t1]+=1
            oracle[t0]-=1
            effect=p1-p0
            effect_error.append(float((effect-oracle).abs().sum()))
            effect_size.append(float(effect.abs().sum()))
            oracle_size.append(float(oracle.abs().sum()))
        mean=lambda values:sum(values)/len(values)
        rows[kind]=dict(n=len(selected),answer_accuracy=mean(correct),
                        answer_ce=mean(ce),effect_to_oracle_l1=mean(effect_error),
                        native_effect_l1=mean(effect_size),
                        oracle_effect_l1=mean(oracle_size))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir',action='append',required=True)
    p.add_argument('--extra-checkpoint',action='append',default=[],
                   help='label=state_dict.pt using the first model directory configuration')
    p.add_argument('--reference-model-dir',default=None,
                   help='Use this model run only to fix the evaluation DGP')
    p.add_argument('--source-seed',type=int,default=88001)
    p.add_argument('--section-seed',type=int,default=88002)
    p.add_argument('--joint-seed',type=int,default=88003)
    p.add_argument('--streams',type=int,default=600)
    p.add_argument('--source-cap',type=int,default=80)
    p.add_argument('--section-cap',type=int,default=120)
    p.add_argument('--joint-streams',type=int,default=256)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config
    saved=[json.loads((Path(folder)/'results.json').read_text())
           for folder in a.model_dir]
    kwargs=(json.loads((Path(a.reference_model_dir)/'results.json').read_text())
            ['generator_kwargs'] if a.reference_model_dir else
            saved[0]['generator_kwargs'])
    source_pairs,source_attempted=source_panel(a.source_seed,kwargs,
                                                a.streams,a.source_cap)
    section_pairs,section_attempted=scope_panel(a.section_seed,kwargs,
                                                  a.streams,a.section_cap)
    assert {p['kind'] for p in source_pairs}=={
        'other_speaker','query_object','statement_speaker'}
    assert {p['kind'] for p in section_pairs}=={
        'section_key','relevant_color','irrelevant_color'}
    generator=SurfaceMixtureV8(seed=a.joint_seed,**kwargs)
    device=torch.device(a.device)
    joint=make_eval(generator,a.joint_streams,256,device)
    vocab=generator.token2id
    rows={}
    model_specs=[(str(Path(folder)),run['records'][0],
                  Path(folder)/f"model_depth{run['records'][0]['depth']}.pt")
                 for folder,run in zip(a.model_dir,saved)]
    for item in a.extra_checkpoint:
        label,path=item.split('=',1)
        if not label or any(label==spec[0] for spec in model_specs):
            p.error(f'Invalid or repeated extra checkpoint label: {label}')
        model_specs.append((label,saved[0]['records'][0],Path(path)))
    for key,record,weights in model_specs:
        assert record['model_config']['vocab_size']==len(vocab)
        assert record['model_config']['seq_len']>=256
        model=BilinearGPT(Config(**record['model_config'])).to(device)
        model.load_state_dict(torch.load(weights,map_location=device,
                                         weights_only=True))
        model.eval().requires_grad_(False)
        with torch.no_grad():
            joint_scores=joint_evaluate(model,joint)
            source_probs,s0,s1=endpoint_predictions(
                model,source_pairs,vocab['.'],device,a.batch_size)
            section_probs,c0,c1=endpoint_predictions(
                model,section_pairs,vocab['.'],device,a.batch_size)
        rows[key]=dict(joint=joint_scores,
                       source=edit_scores(source_pairs,source_probs,s0,s1,vocab),
                       section=edit_scores(section_pairs,section_probs,c0,c1,vocab),
                       parameters=record['parameters'])
        print(key,'source answer KL',round(joint_scores['source_answer']['kl'],3),
              'section answer KL',round(joint_scores['scope_answer']['kl'],3),
              'source query accuracy',round(rows[key]['source']['query_object']
                                            ['answer_accuracy'],3),
              'section key accuracy',round(rows[key]['section']['section_key']
                                            ['answer_accuracy'],3),flush=True)
    result=dict(config=vars(a),reference_generator_kwargs=kwargs,
                source_counts=dict(Counter(x['kind']
                for x in source_pairs)),section_counts=dict(Counter(x['kind']
                for x in section_pairs)),source_attempted=dict(source_attempted),
                section_attempted=dict(section_attempted),models=rows,
                caveat='Fresh exact-oracle and supported-edit evaluation. '
                       'All models trained on these task families; this is a '
                       'competence audit, not a training-task holdout.')
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

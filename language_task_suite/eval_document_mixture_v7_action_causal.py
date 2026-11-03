"""Supported local conditional-action edits for V7-trained models."""

import argparse
import json
import random
from pathlib import Path

import torch

from document_mixture_v5 import COMMON,DIAGNOSTIC
from surface_mixture_v2 import _split
from surface_mixture_v4 import _content
from surface_mixture_v13_attributes import ATTRIBUTES,attribute_state
from surface_mixture_v14_conditional import ACTION,SurfaceMixtureV14Conditional
from train_document_mixture_v5 import DeepModel,SPECS


def collect(seed,streams,length,max_cases):
    gen=SurfaceMixtureV14Conditional(
        seed=seed,conditional_continuation_rate=.35,**COMMON,**DIAGNOSTIC)
    cases=[]
    for _ in range(streams):
        ids,labels,_=gen.sample_sequence(length)
        words=[gen.id2token[i] for i in ids]
        for t,label in enumerate(labels):
            if label!='conditional_action':continue
            history,raw=_split(words[:t])
            content=_content(raw)
            prior=attribute_state(history)['facts']
            old=content[3]
            replacement=next((v for v in ATTRIBUTES
                              if v not in prior.values() and v!=old),None)
            if replacement is None:continue
            position=t-len(raw)+raw.index('The')+3
            changed=words[:t].copy();changed[position]=replacement
            for u in range(position+1,t):
                assert gen.next_distribution(changed[:u]).get(changed[u],0)>0
            answer=ACTION[replacement]
            assert gen.next_distribution(changed)=={answer:1.}
            assert answer!=words[t]
            cases.append((words[:t],changed,answer))
    random.Random(seed+991).shuffle(cases)
    return cases[:max_cases],len(cases)


@torch.no_grad()
def score(model,cases,vocab,n_ctx,device,batch_size):
    responses=[];base=[];edited=[]
    for start in range(0,len(cases),batch_size):
        part=cases[start:start+batch_size]
        before=torch.full((len(part),n_ctx),vocab['<PAD>'],
                          dtype=torch.long,device=device)
        after=before.clone()
        indices=[];targets=[]
        for i,(original,changed,target) in enumerate(part):
            assert len(original)<=n_ctx
            before[i,:len(original)]=torch.tensor(
                [vocab[word] for word in original],device=device)
            after[i,:len(changed)]=torch.tensor(
                [vocab[word] for word in changed],device=device)
            indices.append(len(original)-1)
            targets.append(vocab[target])
        index=torch.tensor(indices,device=device)
        target=torch.tensor(targets,device=device)
        p0=model(before)[torch.arange(len(part),device=device),index]
        p1=model(after)[torch.arange(len(part),device=device),index]
        p0=p0.float().softmax(-1).gather(-1,target[:,None]).flatten()
        p1=p1.float().softmax(-1).gather(-1,target[:,None]).flatten()
        base.extend(p0.cpu().tolist());edited.extend(p1.cpu().tolist())
        responses.extend((p1-p0).cpu().tolist())
    return dict(n=len(cases),mean_model_response=sum(responses)/len(responses),
                mean_abs_response=sum(abs(v) for v in responses)/len(responses),
                mean_target_probability_before=sum(base)/len(base),
                mean_target_probability_after=sum(edited)/len(edited),
                oracle_response=1.)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--streams',type=int,default=400)
    p.add_argument('--length',type=int,default=192)
    p.add_argument('--max-cases',type=int,default=128)
    p.add_argument('--seed',type=int,default=34951)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    vocab=saved['vocabulary'];n_ctx=saved['n_ctx']
    cases,available=collect(a.seed,a.streams,a.length,a.max_cases)
    assert len(cases)==a.max_cases
    args=saved['model_kwargs']
    model=DeepModel(len(vocab),args['d_model'],args['n_head'],
                    SPECS[f"block{saved['depth']}"],n_ctx,norm=args['norm'],
                    residual=args['residual'],attention=args['attention']).to(a.device)
    model.load_state_dict(saved['state_dict']);model.eval()
    result=score(model,cases,vocab,n_ctx,a.device,a.batch_size)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(config={**vars(a),
        'checkpoint':str(a.checkpoint),'output':str(a.output)},
        available_supported_edits=available,result=result,
        caveat='Fresh V14 narrative substream with the same V7 sampling '
               'parameters; edits remain inside V14 support. This gates '
               'the local action rule only, not the cross-section rule.'),
        indent=2)+'\n')
    print('n',result['n'],'response',round(result['mean_model_response'],3),
          flush=True)


if __name__=='__main__':main()

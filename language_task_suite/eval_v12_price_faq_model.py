"""Fresh V12 competence and two supported causal edits at price answers."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
from model import BilinearGPT, Config
from audit_v12_price_faq import fact_positions
from surface_mixture_v2 import _split
from surface_mixture_v4 import _content
from surface_mixture_v12_price_faq import PRICES, SurfaceMixtureV12PriceFAQ, price_state
from train_v12_price_faq import COMMON, ENRICHED, REFERENCE


def collect(seed,nstreams,length,kwargs,max_edits):
    gen=SurfaceMixtureV12PriceFAQ(seed=seed,**COMMON,**kwargs)
    streams=[]
    examples=[]
    for _ in range(nstreams):
        ids,labels,_=gen.sample_sequence(length)
        words=[gen.id2token[v] for v in ids]
        streams.append((ids,labels))
        if len(examples)>=max_edits:
            continue
        positions=fact_positions(words)
        for t,label in enumerate(labels):
            if label!='price_answer' or len(examples)>=max_edits:continue
            history,raw=_split(words[:t])
            content=_content(raw)
            product=content[4]
            state=price_state(history)
            facts=state['facts']
            assert words[t]==facts[product]
            alternatives=[p for p in facts if p!=product and
                          p not in state['answered']]
            replacement=next((v for v in PRICES if v not in facts.values()),None)
            item= dict(x=ids[:-1],query=t-1,product=product,
                       old=words[t],fact_count=len(facts),
                       latest_price=list(facts.values())[-1],
                       mentioned=product in state['mentioned'])
            if replacement is not None:
                pos,_=positions[product]
                edit=words[:t].copy();edit[pos]=replacement
                assert gen.next_distribution(edit)=={replacement:1.}
                for u in range(pos+1,t):
                    assert gen.next_distribution(edit[:u]).get(edit[u],0)>0
                item['fact_edit']=dict(position=pos,new=replacement)
            if alternatives:
                new_product=alternatives[0]
                query_pos=t-len(raw)+raw.index('How')+4
                assert words[query_pos]==product
                assert gen.next_distribution(words[:query_pos]).get(
                    new_product,0)>0
                edit=words[:t].copy();edit[query_pos]=new_product
                assert gen.next_distribution(edit)=={facts[new_product]:1.}
                for u in range(query_pos+1,t):
                    assert gen.next_distribution(edit[:u]).get(edit[u],0)>0
                item['key_edit']=dict(position=query_pos,
                                      new_product=new_product,
                                      new=facts[new_product])
            examples.append(item)
    return gen,streams,examples


@torch.no_grad()
def evaluate(model,gen,streams,examples,device,batch_size):
    totals=defaultdict(list)
    for start in range(0,len(streams),batch_size):
        part=streams[start:start+batch_size]
        ids=torch.tensor([row[0] for row in part],device=device)
        logp=model(ids[:,:-1])[0].float().log_softmax(-1)
        ce=-logp.gather(-1,ids[:,1:,None]).squeeze(-1).cpu().numpy()
        for b,(_,labels) in enumerate(part):
            for t,label in enumerate(labels[1:]):
                totals[label].append(float(ce[b,t]))
    scored=[]
    for start in range(0,len(examples),batch_size):
        part=examples[start:start+batch_size]
        base=torch.tensor([v['x'] for v in part],device=device)
        altered=[];metadata=[]
        for i,item in enumerate(part):
            for kind in ('fact_edit','key_edit'):
                if kind not in item:continue
                row=base[i].clone()
                edit=item[kind]
                row[edit['position']]=gen.token2id[
                    edit.get('new_product',edit['new'])]
                altered.append(row)
                metadata.append((i,kind,edit['new']))
        query=torch.tensor([v['query'] for v in part],device=device)
        base_p=model(base)[0].float().softmax(-1)[
            torch.arange(len(part),device=device),query]
        changed_p=(model(torch.stack(altered))[0].float().softmax(-1)[
            torch.arange(len(altered),device=device),
            torch.tensor([part[i]['query'] for i,_,_ in metadata],device=device)]
            if altered else None)
        for i,item in enumerate(part):
            scored.append(dict(product=item['product'],old=item['old'],
                fact_count=item['fact_count'],mentioned=item['mentioned'],
                correct_probability=float(base_p[i,gen.token2id[item['old']]]),
                latest_fact_correct=item['latest_price']==item['old']))
        for j,(i,kind,new) in enumerate(metadata):
            out=scored[start+i]
            out[kind]=dict(new=new,
                before=float(base_p[i,gen.token2id[new]]),
                after=float(changed_p[j,gen.token2id[new]]),
                response=float(changed_p[j,gen.token2id[new]]-
                               base_p[i,gen.token2id[new]]),
                edited_top=gen.id2token[int(changed_p[j].argmax())])
    summary={name:dict(n=len(values),ce=float(np.mean(values)))
             for name,values in totals.items()}
    edits={}
    for kind in ('fact_edit','key_edit'):
        group=[row[kind] for row in scored if kind in row]
        edits[kind]=dict(n=len(group),mean_response=float(np.mean(
            [v['response'] for v in group])) if group else None,
            mean_edited_target_probability=float(np.mean(
                [v['after'] for v in group])) if group else None,
            edited_top_accuracy=float(np.mean([
                v['edited_top']==v['new'] for v in group])) if group else None,
            oracle_response=1.)
    return dict(per_rule=summary,edits=edits,
        price_answer_baselines=dict(
            latest_fact_accuracy=float(np.mean([
                v['latest_fact_correct'] for v in scored])) if scored else None,
            product_only_accuracy=1/len(PRICES),
            caveat='Product-only baseline is the balanced prior implied by '
                   'uniform without-replacement price assignment; latest-fact '
                   'accuracy is measured on this panel.'),
        answer_examples=len(scored),
        answer_with_narrative_mention=sum(v['mentioned'] for v in scored))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=99701)
    parser.add_argument('--streams',type=int,default=512)
    parser.add_argument('--length',type=int,default=256)
    parser.add_argument('--max-edits',type=int,default=128)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    saved=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
    assert saved['generator']=='SurfaceMixtureV12PriceFAQ'
    model=BilinearGPT(Config(**saved['config'])).to(args.device)
    model.load_state_dict(saved['model'])
    model.eval()
    panels={}
    for name,kwargs in (('enriched',ENRICHED),('reference_candidate',REFERENCE)):
        gen,streams,examples=collect(args.seed,args.streams,args.length,
                                     kwargs,args.max_edits)
        panels[name]=evaluate(model,gen,streams,examples,
                              torch.device(args.device),args.batch_size)
        print(name,'answer',panels[name]['per_rule'].get('price_answer'),
              'edits',panels[name]['edits'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(
        config={**vars(args),'checkpoint':str(args.checkpoint),
                'output':str(args.output)},
        panels=panels,caveat='Exact generator oracle is deterministic at price '
            'answers. Edit responses are probabilities of the new exact target. '
            'Model competence is assessed separately from DGP support and '
            'real-language similarity.'),indent=2)+'\n')


if __name__=='__main__':main()

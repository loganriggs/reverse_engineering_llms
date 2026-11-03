"""Reference-rate oracle and supported heading-edit evaluation for V11."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0,'/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')

from audit_v11_section_topic import REFERENCE,latest_heading_topic_index
from surface_mixture_v7 import TOPICS
from surface_mixture_v11_section_topic import SurfaceMixtureV11SectionTopic
from train_surface_mixture import evaluate,make_eval
from model import BilinearGPT,Config


def selected_pairs(gen,streams,length,limit,selection_seed):
    pairs=[]
    choice_rng=random.Random(selection_seed)
    for _ in range(streams):
        ids,labels,_=gen.sample_sequence(length)
        words=[gen.id2token[i] for i in ids]
        for t,label in enumerate(labels):
            if label!='section_topic_place' or t==0 or t>=length-1:
                continue
            prefix=words[:t]
            heading=latest_heading_topic_index(prefix)
            assert heading is not None
            old=prefix[heading]
            base=gen.place_rule_components(prefix)['base_place']['distribution']
            alternatives=[topic for topic in TOPICS if topic!=old and topic in base]
            if not alternatives:continue
            new=choice_rng.choice(alternatives)
            edited=prefix.copy();edited[heading]=new
            for pos in range(heading+1,len(prefix)):
                assert gen.next_distribution(edited[:pos]).get(edited[pos],0)>0
            old_q=gen.next_distribution(prefix)
            new_q=gen.next_distribution(edited)
            assert abs(new_q[new]-old_q[new]-gen.section_topic_mass)<1e-10
            pairs.append(dict(original=ids[:-1],
                              edited=ids[:heading]+[gen.token2id[new]]+
                                     ids[heading+1:-1],
                              query=t-1,old_topic=old,new_topic=new,
                              old_q=old_q,new_q=new_q,
                              heading_distance=t-heading))
    if len(pairs)<limit:
        raise ValueError(f'Only {len(pairs)} supported edits in {streams} streams')
    return choice_rng.sample(pairs,limit)


@torch.no_grad()
def score_pairs(model,gen,pairs,batch_size,device):
    rows=[]
    vocab=gen.vocab
    for start in range(0,len(pairs),batch_size):
        part=pairs[start:start+batch_size]
        original=torch.tensor([row['original'] for row in part],device=device)
        edited=torch.tensor([row['edited'] for row in part],device=device)
        q=torch.tensor([row['query'] for row in part],device=device)
        combined=torch.cat((original,edited))
        logits=model(combined)[0].float()[
            torch.arange(len(combined),device=device),torch.cat((q,q))]
        probs=logits.softmax(-1)
        logp=logits.log_softmax(-1)
        n=len(part)
        for i,row in enumerate(part):
            target=gen.token2id[row['new_topic']]
            target_old=torch.tensor([row['old_q'].get(w,0.) for w in vocab],device=device)
            target_new=torch.tensor([row['new_q'].get(w,0.) for w in vocab],device=device)
            response=float(probs[n+i,target]-probs[i,target])
            oracle=float(target_new[target]-target_old[target])
            effect=(probs[n+i]-probs[i])-(target_new-target_old)
            def kl(target_dist,log_prob):
                return float((target_dist*(target_dist.clamp_min(1e-30).log()-log_prob)).sum())
            rows.append(dict(response=response,oracle_response=oracle,
                             effect_l1_error=float(effect.abs().sum()),
                             original_kl=kl(target_old,logp[i]),
                             edited_kl=kl(target_new,logp[n+i]),
                             heading_distance=row['heading_distance']))
    def summarize(subset):
        return dict(n=len(subset),**{key:sum(row[key] for row in subset)/len(subset)
            for key in subset[0]})
    by_distance={}
    for name,condition in [('near',lambda d:d<=16),
                           ('middle',lambda d:16<d<=64),
                           ('far',lambda d:d>64)]:
        selected=[row for row in rows if condition(row['heading_distance'])]
        if selected:by_distance[name]=summarize(selected)
    return dict(all=summarize(rows),by_distance=by_distance)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pretrained',type=Path,required=True)
    p.add_argument('--trained',type=Path,required=True)
    p.add_argument('--seed',type=int,default=98501)
    p.add_argument('--streams',type=int,default=512)
    p.add_argument('--pairs',type=int,default=256)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--section-rate',type=float,default=REFERENCE['section_rate'])
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    device=torch.device(a.device)
    kwargs=dict(REFERENCE,section_rate=a.section_rate,section_topic_mass=.35)
    gen=SurfaceMixtureV11SectionTopic(seed=a.seed,**kwargs)
    data=make_eval(gen,a.streams,a.length,device)
    pairs=selected_pairs(SurfaceMixtureV11SectionTopic(seed=a.seed+1,**kwargs),
                         a.streams,a.length+1,a.pairs,a.seed+2)
    assert len(pairs)==a.pairs
    assert int(data['section_topic_place'].sum())>0
    # Both checkpoints have the same vocabulary and bilinear architecture.
    config=Config(vocab_size=len(gen.vocab),n_layer=2,n_head=4,n_embd=64,
                  seq_len=a.length,block_has_mlp=(False,False),
                  use_rmsnorm=True)
    results={}
    for name,path in [('pretrained_v10',a.pretrained),('trained_v11',a.trained)]:
        model=BilinearGPT(config).to(device)
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
        model.eval()
        ordinary=evaluate(model,data)
        causal=score_pairs(model,gen,pairs,32,device)
        results[name]=dict(ordinary=ordinary,heading_edit=causal)
        print(name,'topic KL',round(ordinary['section_topic_place']['kl'],4),
              'edit',round(causal['all']['response'],4),
              'oracle',round(causal['all']['oracle_response'],4),
              'effect L1',round(causal['all']['effect_l1_error'],4),flush=True)
    output=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                reference_generator_kwargs=kwargs,
                topic_site_count=int(data['section_topic_place'].sum()),
                results=results,
                caveat='The pretrained V10 model is evaluated on a new V11 '
                       'reference-rate DGP; the V11 checkpoint may have been '
                       'trained on a higher section rate. The supported edit '
                       'changes a heading topic while keeping all intervening '
                       'tokens fixed. This does not establish natural-language '
                       'realism or whole-task model-training generalization.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()

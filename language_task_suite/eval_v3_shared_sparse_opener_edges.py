"""Does the sparse opener edge mediate a supported bracket-type edit?"""

import argparse
import json
from pathlib import Path

import torch

from eval_original_v3_shared_bracket import selected_pairs
from eval_v3_shared_sparse_rule_edges import ablated_logits
from original_v3_shared_bracket import LanguageV3SharedBracket
from model import BilinearGPT, Config


@torch.no_grad()
def evaluate(model,gen,pairs,head,control,batch_size,device):
    rows=[]
    for start in range(0,len(pairs),batch_size):
        part=pairs[start:start+batch_size]
        original=torch.tensor([row['original'] for row in part],device=device)
        edited=torch.tensor([row['edited'] for row in part],device=device)
        combined=torch.cat((original,edited))
        q=torch.tensor([row['query'] for row in part],device=device)
        opener=torch.tensor([(original[i]!=edited[i]).nonzero().flatten().item()
                             for i in range(len(part))],device=device)
        k=opener+(control=='next_token')
        assert (k<=q).all()
        both_q=torch.cat((q,q))
        both_k=torch.cat((k,k))
        idx=torch.arange(len(combined),device=device)
        native=model(combined)[0][idx,both_q].float().softmax(-1)
        ablated=ablated_logits(model,combined,both_q,both_k,head)[
            idx,both_q].float().softmax(-1)
        n=len(part)
        for i,row in enumerate(part):
            new=gen.token2id[row['new_closer']]
            old=gen.token2id[row['old_closer']]
            oracle=(row['new_oracle'].get(row['new_closer'],0.)-
                    row['old_oracle'].get(row['new_closer'],0.))
            for name,prob in (('native',native),('ablated',ablated)):
                response=float(prob[n+i,new]-prob[i,new])
                full_effect=(prob[n+i]-prob[i])
                expected=torch.tensor([
                    row['new_oracle'].get(token,0.)-
                    row['old_oracle'].get(token,0.) for token in gen.vocab],
                    device=device)
                rows.append(dict(pair=start+i,variant=name,
                                 response=response,oracle_response=oracle,
                                 effect_l1_error=float((full_effect-expected).abs().sum()),
                                 old_closer_before=float(prob[i,old]),
                                 new_closer_after=float(prob[n+i,new])))
    result={}
    for variant in ('native','ablated'):
        subset=[row for row in rows if row['variant']==variant]
        result[variant]=dict(n=len(subset),**{key:sum(row[key] for row in subset)/len(subset)
            for key in ('response','oracle_response','effect_l1_error',
                        'old_closer_before','new_closer_after')})
    result['response_change']=result['ablated']['response']-result['native']['response']
    result['effect_l1_error_change']=(result['ablated']['effect_l1_error']-
                                     result['native']['effect_l1_error'])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sparse-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=98101)
    p.add_argument('--streams',type=int,default=512)
    p.add_argument('--pairs',type=int,default=256)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    torch.set_num_threads(1)
    device=torch.device(a.device)
    gen=LanguageV3SharedBracket(seed=a.seed)
    pairs=selected_pairs(gen,a.streams,129,a.pairs)
    assert len(pairs)==a.pairs
    results={}
    for name,checkpoint in [('dense','dense.pt'),('sparse60','final_no_l1.pt')]:
        payload=torch.load(a.sparse_run/checkpoint,map_location=device,
                           weights_only=True)
        model=BilinearGPT(Config(**payload['config'])).to(device)
        model.load_state_dict(payload['model'])
        model.eval()
        results[name]={}
        for head in range(4):
            for control in ('opener','next_token'):
                row=evaluate(model,gen,pairs,head,control,32,device)
                results[name][f'H{head}_{control}']=row
                print(name,head,control,'native',round(row['native']['response'],3),
                      'ablated',round(row['ablated']['response'],3),
                      'oracle',round(row['native']['oracle_response'],3),flush=True)
    output=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                results=results,
                caveat='An opener swap retains the intervening content prefix '
                       'in both bracket branches of the DGP. This removes one '
                       'layer-1 head edge in both original and edited model '
                       'runs. It tests local causal mediation, not a unique '
                       'minimal circuit or natural-language generalization.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()

"""Fresh exact-oracle and supported-edit evaluation of a trained V10 model."""

import argparse
import json
import sys
from pathlib import Path

import torch

from surface_mixture_v10_linked_events import SurfaceMixtureV10LinkedEvents
from surface_mixture_v10_linked_events import linked_state
from surface_mixture_v2 import _split
from surface_mixture_v4 import _content
from train_surface_mixture import evaluate, make_eval


@torch.no_grad()
def chosen_probabilities(model, rows, positions, tokens, device):
    values=[]
    for start in range(0,len(rows),32):
        batch=torch.stack(rows[start:start+32]).to(device)
        pos=torch.tensor(positions[start:start+32],device=device)
        tok=torch.tensor(tokens[start:start+32],device=device)
        p=model(batch)[0][torch.arange(len(batch),device=device),pos].softmax(-1)
        values.extend(p.gather(-1,tok[:,None]).squeeze(-1).cpu().tolist())
    return values


def bracket_pairs(gen,data,limit):
    original,edited,positions,old,new,oracle_mass=[],[],[],[],[],[]
    x=data['x'].cpu()
    for b,t in torch.nonzero(data['bracket_close'],as_tuple=False).tolist():
        words=[gen.id2token[int(i)] for i in x[b,:t+1]]
        matches=[k for k in range(t,-1,-1) if words[k] in ('(','[')
                 and '.' not in words[k:t+1]]
        if not matches: continue
        k=matches[0]
        old_close=')' if words[k]=='(' else ']'
        new_close=']' if old_close==')' else ')'
        assert gen.next_distribution(words[:k]).get('(',0)>0
        assert gen.next_distribution(words[:k]).get('[',0)>0
        edited_words=words.copy()
        edited_words[k]='[' if words[k]=='(' else '('
        before=gen.next_distribution(words)
        after=gen.next_distribution(edited_words)
        assert before.get(old_close,0)>0 and after.get(new_close,0)==before[old_close]
        assert after.get(old_close,0)==0
        assert all(gen.next_distribution(edited_words[:j]).get(edited_words[j],0)>0
                   for j in range(k,len(edited_words)))
        original.append(x[b].clone())
        newrow=x[b].clone(); newrow[k]=gen.token2id[edited_words[k]]
        edited.append(newrow)
        positions.append(t); old.append(gen.token2id[old_close]);new.append(gen.token2id[new_close])
        oracle_mass.append(before[old_close])
        if len(original)>=limit: break
    return original,edited,positions,old,new,oracle_mass


def source_pairs(gen,data,limit):
    original,edited,positions,old,new=[],[],[],[],[]
    x=data['x'].cpu()
    for b,t in torch.nonzero(data['source_answer'],as_tuple=False).tolist():
        words=[gen.id2token[int(i)] for i in x[b,:t+1]]
        key_pos=t-5
        before=gen.next_distribution(words)
        assert len(before)==1
        old_answer=next(iter(before))
        for key in gen.next_distribution(words[:key_pos]):
            if key==words[key_pos]:continue
            changed=words.copy();changed[key_pos]=key
            after=gen.next_distribution(changed)
            if len(after)!=1 or old_answer in after:continue
            assert all(gen.next_distribution(changed[:j]).get(changed[j],0)>0
                       for j in range(key_pos,len(changed)))
            original.append(x[b].clone())
            newrow=x[b].clone();newrow[key_pos]=gen.token2id[key]
            edited.append(newrow)
            positions.append(t);old.append(gen.token2id[old_answer])
            new.append(gen.token2id[next(iter(after))])
            break
        if len(original)>=limit:break
    return original,edited,positions,old,new


def source_claim_pairs(gen,data,limit):
    """Supported earlier claim-object edit at a pending linked prose object slot."""
    original,edited,positions,old,new,oracle_effect=[],[],[],[],[],[]
    x=data['x'].cpu()
    for b,t in torch.nonzero(data['linked_object'],as_tuple=False).tolist():
        words=[gen.id2token[int(i)] for i in x[b,:t+1]]
        history,_=_split(words)
        pending=linked_state(history)['pending']
        if pending is None:continue
        obj,speaker=pending
        offset=0
        claim_pos=None
        for raw in history:
            content=_content(raw)
            if content and content[0]=='As' and content[1]==speaker and obj in content:
                claim_pos=offset+raw.index(obj)
            offset+=len(raw)
        if claim_pos is None:continue
        before=gen.next_distribution(words)
        allowed=gen.next_distribution(words[:claim_pos])
        for alt in allowed:
            if alt==obj or alt in linked_state(history)['claims']:
                continue
            changed=words.copy();changed[claim_pos]=alt
            try:
                if not all(gen.next_distribution(changed[:j]).get(changed[j],0)>0
                           for j in range(claim_pos,len(changed))):
                    continue
                after=gen.next_distribution(changed)
            except (AssertionError,ValueError,KeyError):
                continue
            if after.get(alt,0)<=before.get(alt,0):continue
            original.append(x[b].clone())
            newrow=x[b].clone();newrow[claim_pos]=gen.token2id[alt]
            edited.append(newrow)
            positions.append(t);old.append(gen.token2id[obj]);new.append(gen.token2id[alt])
            oracle_effect.append(after.get(alt,0)-before.get(alt,0))
            break
        if len(original)>=limit:break
    return original,edited,positions,old,new,oracle_effect


def edit_scores(model,gen,data,device,limit):
    bracket=bracket_pairs(gen,data,limit)
    source=source_pairs(gen,data,limit)
    claim=source_claim_pairs(gen,data,limit)
    out={}
    for name,pair in [('bracket',bracket),('source_query_key',source),
                      ('source_claim_object',claim)]:
        original,edited,positions,old,new=pair[:5]
        if not original:
            out[name]=dict(n=0)
            continue
        old_before=chosen_probabilities(model,original,positions,old,device)
        old_after=chosen_probabilities(model,edited,positions,old,device)
        new_before=chosen_probabilities(model,original,positions,new,device)
        new_after=chosen_probabilities(model,edited,positions,new,device)
        avg=lambda xs:sum(xs)/len(xs)
        record=dict(n=len(original),old_answer_before=avg(old_before),
                    old_answer_after=avg(old_after),
                    new_answer_before=avg(new_before),
                    new_answer_after=avg(new_after),
                    mean_new_answer_effect=avg([b-a for a,b in zip(new_before,new_after)]))
        if name=='bracket':
            record['oracle_mean_closer_effect']=avg(pair[5])
        if name=='source_claim_object':
            record['oracle_mean_new_object_effect']=avg(pair[5])
        out[name]=record
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--streams',type=int,default=256)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--seed',type=int,default=99001)
    p.add_argument('--edit-limit',type=int,default=128)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config
    run=json.loads((Path(a.run_dir)/'results.json').read_text())
    assert run['config']['generator']=='v10'
    result=[]
    for record in run['records']:
        model=BilinearGPT(Config(**record['model_config'])).to(a.device)
        model.load_state_dict(torch.load(Path(a.run_dir)/f"model_depth{record['depth']}.pt",
                                         map_location=a.device,weights_only=True))
        model.eval()
        gen=SurfaceMixtureV10LinkedEvents(seed=a.seed,**run['generator_kwargs'])
        data=make_eval(gen,a.streams,a.length,a.device)
        result.append(dict(depth=record['depth'],scores=evaluate(model,data),
                           supported_edits=edit_scores(model,gen,data,a.device,a.edit_limit)))
    output=dict(config=vars(a),run_config=run['config'],results=result,
                caveat='Fresh sampled DGP, same rule family and vocabulary as training. '
                       'Supported edits measure model response but do not localize circuits. '
                       'No whole-task model-training holdout or natural-language realism claim.')
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2)+'\n')
    for row in result:
        print('depth',row['depth'],{k:row['scores'].get(k)
              for k in ('verb','object','source_influenced_object',
                        'pending_claim_matching_object','source_answer',
                        'scope_answer','bracket_close')})
        print('edits',row['supported_edits'])


if __name__=='__main__':main()

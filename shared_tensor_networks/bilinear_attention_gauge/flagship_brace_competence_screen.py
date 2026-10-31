"""Native-model brace-closure competence screen on code-like prefixes."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_rank_pilot import final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


# Each tuple is (family, brace prompt, alternative prompt, alternative closer).
CANDIDATES=(
    ('set_list','items = {1, 2, 3','items = [1, 2, 3',']'),
    ('set_list','values = {4, 5, 6','values = [4, 5, 6',']'),
    ('set_list','x = {2, 3','x = [2, 3',']'),
    ('set_list','return {1, 2','return [1, 2',']'),
    ('set_list','data = {"x", "y"','data = ["x", "y"',']'),
    ('set_list','x = {','x = [',']'),
    ('set_list','return {','return [',']'),
    ('set_list','items = {','items = [',']'),
    ('set_list','const x = {','const x = [',']'),
    ('set_list','{','[',']'),
    ('set_list','items = { 1, 2','items = [ 1, 2',']'),
    ('set_list','items = { "a", "b"','items = [ "a", "b"',']'),
    ('set_list','nums = {0, 1','nums = [0, 1',']'),
    ('set_list','colors = {"red", "blue"','colors = ["red", "blue"',']'),
    ('set_list','S = {0, 1','S = [0, 1',']'),
    ('tuple_set','items = {1, 2, 3','items = (1, 2, 3',')'),
    ('tuple_set','x = {2, 3','x = (2, 3',')'),
    ('tuple_set','return {1, 2','return (1, 2',')'),
    ('tuple_set','S = {1, 2, 3','S = (1, 2, 3',')'),
)


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    closer=tokenizer.encode('}',add_special_tokens=False).ids
    assert len(closer)==1
    brace_closer=closer[0]
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    rows=[]
    with torch.no_grad():
        for family,brace_text,other_text,other_closer_text in CANDIDATES:
            brace_ids=tokenizer.encode(brace_text,add_special_tokens=False).ids
            other_ids=tokenizer.encode(other_text,add_special_tokens=False).ids
            changed=[(i,x,y) for i,(x,y) in enumerate(zip(brace_ids,other_ids)) if x!=y]
            assert (len(brace_ids)==len(other_ids) and len(changed)==1),(
                brace_text,other_text,brace_ids,other_ids)
            other_closer=tokenizer.encode(other_closer_text,add_special_tokens=False).ids
            assert len(other_closer)==1
            alt=other_closer[0]
            a=final_logits(model,torch.tensor(brace_ids)[None,:])[0]
            b=final_logits(model,torch.tensor(other_ids)[None,:])[0]
            rows.append(dict(family=family,brace_text=brace_text,
                alternative_text=other_text,brace_ids=brace_ids,
                alternative_ids=other_ids,token_swap=changed[0][1:],
                alternative_closer=other_closer_text,
                brace_closer_margin=float(a[brace_closer]-a[alt]),
                alternative_closer_margin=float(b[alt]-b[brace_closer]),
                brace_closer_rank=int((a>a[brace_closer]).sum())+1,
                alternative_closer_rank=int((b>b[alt]).sum())+1,
                brace_closer_probability=float(torch.softmax(a,dim=0)[brace_closer]),
                alternative_closer_probability=float(torch.softmax(b,dim=0)[alt])))
    summary=[]
    for family in ('set_list','tuple_set'):
        group=[r for r in rows if r['family']==family]
        summary.append(dict(family=family,n=len(group),
            brace_forced_choice_correct=sum(r['brace_closer_margin']>0 for r in group),
            alternative_forced_choice_correct=sum(
                r['alternative_closer_margin']>0 for r in group),
            brace_rank_median=float(np.median([r['brace_closer_rank'] for r in group])),
            brace_rank_min=min(r['brace_closer_rank'] for r in group),
            brace_rank_max=max(r['brace_closer_rank'] for r in group),
            brace_margin_mean=float(np.mean([r['brace_closer_margin'] for r in group]))))
    result=dict(description='Exploratory native-model screen of code-like brace versus square/parenthesis closure; every prompt pair differs by one input token, no gauge selected or fitted',
        brace_closer_token_id=brace_closer,n_cases=len(rows),summary=summary,rows=rows)
    (ART/'brace_competence_screen.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in summary:print(row,flush=True)


if __name__=='__main__':run()

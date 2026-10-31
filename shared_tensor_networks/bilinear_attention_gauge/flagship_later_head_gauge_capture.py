"""Capture edit vectors and local margin gradients in two fixed rank-4 cuts."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_head_holdout import pairs as holdout_pairs
from flagship_explicit_task_path_patch import proposed_pairs
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_rank_pilot import HeadCut, encode_pairs, final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

TARGETS={'bracket_closure':(13,8),'subject_verb_agreement':(11,3)}


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probes={layer:HeadCut(model.transformer.h[layer].attn)
            for layer in sorted(set(layer for layer,_ in TARGETS.values()))}
    bases=np.load(ART/'later_head_rank_pilot_bases.npz')
    panels={'train':encode_pairs(tokenizer,proposed_pairs()),
            'holdout':encode_pairs(tokenizer,holdout_pairs())}
    arrays={};meta={}
    gradient_gates={}
    for target,(layer,head) in TARGETS.items():
        probe=probes[layer];probe.head=head
        basis=torch.tensor(bases[f'{target}__weight'][:,:4].copy(),dtype=torch.float32)
        probe.basis=basis
        for panel,examples in panels.items():
            d_rows=[];g_rows=[];raw_d_rows=[];rows=[]
            for family,before,after,answer_a,answer_b in examples:
                probe.capture_grad=True
                val=final_logits(model,torch.tensor(before,dtype=torch.long)[None,:])[0]
                margin=val[answer_a]-val[answer_b]
                x=probe.last_input
                a=head*probe.head_dim
                base=x.detach()[0,-1,a:a+probe.head_dim].clone()
                grad=torch.autograd.grad(margin,x)[0][0,-1,a:a+probe.head_dim].detach()
                probe.capture_grad=False
                with torch.no_grad():
                    final_logits(model,torch.tensor(after,dtype=torch.long)[None,:])
                    edited=probe.last_input[0,-1,a:a+probe.head_dim].clone()
                delta=edited-base
                d_rows.append((delta@basis).numpy())
                g_rows.append((grad@basis).numpy())
                raw_d_rows.append(delta.numpy())
                rows.append(dict(family=family,before_ids=before,after_ids=after,
                                 answer_a=answer_a,answer_b=answer_b,
                                 base_margin=float(margin.detach())))
            arrays[f'{target}__{panel}__d']=np.stack(d_rows)
            arrays[f'{target}__{panel}__g']=np.stack(g_rows)
            arrays[f'{target}__{panel}__raw_delta']=np.stack(raw_d_rows)
            meta[f'{target}__{panel}']=rows
            print(target,panel,len(rows),'gradient norms',
                  float(np.mean(np.linalg.norm(np.stack(g_rows),axis=1))),flush=True)
        probe.basis=None
        item=meta[f'{target}__train'][0]
        probe.basis=basis
        direction=basis[:,0][None,:]
        with torch.no_grad():
            probe.injection=.01*direction
            plus=final_logits(model,torch.tensor(item['before_ids'])[None,:])[0]
            plus=float(plus[item['answer_a']]-plus[item['answer_b']])
            probe.injection=-.01*direction
            minus=final_logits(model,torch.tensor(item['before_ids'])[None,:])[0]
            minus=float(minus[item['answer_a']]-minus[item['answer_b']])
            probe.injection=None
        finite=(plus-minus)/.02
        analytic=float(arrays[f'{target}__train__g'][0,0])
        gradient_gates[target]=dict(analytic=analytic,finite=finite,
                                    absolute_error=abs(finite-analytic))
        assert abs(finite-analytic)<1e-3,gradient_gates[target]
        probe.basis=None
    np.savez(ART/'later_head_gauge_capture.npz',**arrays)
    (ART/'later_head_gauge_capture.json').write_text(json.dumps(dict(
        description='Rank-4 weight-basis edit coordinates and local margin gradients; 23 discovery and 21 heldout pairs',
        targets=TARGETS,gradient_gates=gradient_gates,rows=meta),indent=2)+'\n')


if __name__=='__main__':run()

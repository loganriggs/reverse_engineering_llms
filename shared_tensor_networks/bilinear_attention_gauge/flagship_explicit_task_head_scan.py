"""Locate attention heads carrying natural edit effects at layers 8, 11, 13, 14."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_path_patch import proposed_pairs, score
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT, GPTConfig
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

LAYERS = (8, 11, 13, 14)


class ProjectionInputProbe:
    def __init__(self, attention):
        self.native = attention.c_proj.forward
        self.head_dim = attention.head_dim
        self.n_head = attention.n_head
        self.last_input = None
        self.injection = None
        attention.c_proj.forward = self.forward

    def forward(self, x):
        self.last_input = x.detach().clone()
        if self.injection is not None:
            head, delta = self.injection
            x = x.clone()
            a = head * self.head_dim
            x[:, :, a:a+self.head_dim] += delta
        return self.native(x)


def run():
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(TOKENIZER)
    config = json.loads((SNAPSHOT / 'config.json').read_text())
    config.pop('step', None)
    model = GPT(GPTConfig(**config)).eval()
    state = torch.load(SNAPSHOT / 'pytorch_model.bin', map_location='cpu',
                       weights_only=True, mmap=True)
    model.load_state_dict(state)
    del state
    model.requires_grad_(False)
    probes = {layer: ProjectionInputProbe(model.transformer.h[layer].attn)
              for layer in LAYERS}
    rows = []
    with torch.no_grad():
        for family, base_prompt, edited_prompt, answer_a, answer_b in proposed_pairs():
            before = tokenizer.encode(base_prompt, add_special_tokens=False).ids
            after = tokenizer.encode(edited_prompt, add_special_tokens=False).ids
            choices = [tokenizer.encode(x, add_special_tokens=False).ids
                       for x in (answer_a, answer_b)]
            assert len(before) == len(after)
            assert len(choices[0]) == len(choices[1]) == 1
            assert sum(a != b for a, b in zip(before, after)) == 1
            base = score(model, before, choices[0][0], choices[1][0])
            base_inputs = {layer: p.last_input.clone() for layer,p in probes.items()}
            edited = score(model, after, choices[0][0], choices[1][0])
            edited_inputs = {layer: p.last_input.clone() for layer,p in probes.items()}
            effects = {}
            for layer, probe in probes.items():
                delta = edited_inputs[layer] - base_inputs[layer]
                head_effects = []
                for head in range(probe.n_head):
                    a = head * probe.head_dim
                    probe.injection = (head, delta[:,:,a:a+probe.head_dim])
                    head_effects.append(score(model, before, choices[0][0], choices[1][0])-base)
                    probe.injection = None
                effects[str(layer)] = head_effects
            rows.append(dict(family=family, base_prompt=base_prompt,
                             edited_prompt=edited_prompt,
                             native_edit_effect=edited-base,
                             head_effects=effects,
                             both_choices_correct=base>0 and edited<0))
            print(family,len(rows),flush=True)
    summary={}
    for family in sorted(set(r['family'] for r in rows)):
        part=[r for r in rows if r['family']==family]
        native=np.array([r['native_edit_effect'] for r in part])
        layers={}
        for layer in LAYERS:
            e=np.array([r['head_effects'][str(layer)] for r in part])
            layers[str(layer)]=dict(mean_abs_by_head=np.mean(np.abs(e),axis=0).tolist(),
                                   mean_signed_by_head=np.mean(e,axis=0).tolist(),
                                   sign_alignment_by_head=np.mean(np.sign(e)==np.sign(native[:,None]),axis=0).tolist())
        summary[family]=dict(n=len(part),both_choices_correct=sum(r['both_choices_correct'] for r in part),
                             mean_abs_native=float(np.mean(np.abs(native))),layers=layers)
    result=dict(description='Edited-minus-base pre-projection attention vector patched one head at a time; path sufficiency only',
                summary=summary,rows=rows)
    path=ART/'all_head_explicit_task_head_attention_scan.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',path,flush=True)
    return result


if __name__=='__main__':
    run()

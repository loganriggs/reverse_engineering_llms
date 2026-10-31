"""Scan attention-output paths for competent natural token-swap tasks.

For each equal-length single-token pair, transplant one layer's complete
attention output from the edited run into the base run. This is a sufficiency
test for that path, not an additive mediation decomposition.
"""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_path_patch import proposed_pairs, score
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT, GPTConfig
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


class AttentionOutputProbe:
    def __init__(self, attention):
        self.native = attention.forward
        self.last_output = None
        self.delta = None
        attention.forward = self.forward

    def forward(self, x, v1=None):
        output, carried = self.native(x, v1)
        self.last_output = output.detach().clone()
        if self.delta is not None:
            output = output + self.delta
        return output, carried


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
    probes = [AttentionOutputProbe(block.attn) for block in model.transformer.h]
    rows = []
    with torch.no_grad():
        for family, base_prompt, edited_prompt, answer_a, answer_b in proposed_pairs():
            before = tokenizer.encode(base_prompt, add_special_tokens=False).ids
            after = tokenizer.encode(edited_prompt, add_special_tokens=False).ids
            choices = [tokenizer.encode(x, add_special_tokens=False).ids
                       for x in (answer_a, answer_b)]
            assert len(choices[0]) == len(choices[1]) == 1
            assert len(before) == len(after)
            changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
            assert len(changed) == 1, (base_prompt, edited_prompt)
            base_margin = score(model, before, choices[0][0], choices[1][0])
            base_outputs = [p.last_output.clone() for p in probes]
            edited_margin = score(model, after, choices[0][0], choices[1][0])
            edited_outputs = [p.last_output.clone() for p in probes]
            layer_effects = []
            for layer, probe in enumerate(probes):
                probe.delta = edited_outputs[layer] - base_outputs[layer]
                patched_margin = score(model, before, choices[0][0], choices[1][0])
                probe.delta = None
                layer_effects.append(patched_margin - base_margin)
            rows.append(dict(family=family, base_prompt=base_prompt,
                             edited_prompt=edited_prompt,
                             changed_position=changed[0], base_margin=base_margin,
                             edited_margin=edited_margin,
                             native_edit_effect=edited_margin-base_margin,
                             layer_attention_effects=layer_effects,
                             both_choices_correct=(base_margin > 0 and edited_margin < 0)))
            print(family, len(rows), 'native', round(edited_margin-base_margin, 3),
                  'max_attention', round(max(layer_effects, key=abs), 3),
                  'layer', int(np.argmax(np.abs(layer_effects))), flush=True)
    summary = {}
    for family in sorted(set(r['family'] for r in rows)):
        subset = [r for r in rows if r['family'] == family]
        effects = np.array([r['layer_attention_effects'] for r in subset])
        native = np.array([r['native_edit_effect'] for r in subset])
        summary[family] = dict(n=len(subset),
            both_choices_correct=int(sum(r['both_choices_correct'] for r in subset)),
            mean_abs_native_edit=float(np.mean(np.abs(native))),
            mean_abs_attention_by_layer=np.mean(np.abs(effects), axis=0).tolist(),
            signed_alignment_by_layer=np.mean(np.sign(effects) == np.sign(native[:,None]), axis=0).tolist(),
            max_mean_abs_attention_layer=int(np.argmax(np.mean(np.abs(effects), axis=0))))
    result = dict(description='Native model, edited-minus-base full attention output transplanted at one layer at a time; path sufficiency only',
                  summary=summary, rows=rows)
    path = ART / 'all_head_explicit_task_layer_attention_scan.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print('Saved', path, flush=True)
    return result


if __name__ == '__main__':
    run()

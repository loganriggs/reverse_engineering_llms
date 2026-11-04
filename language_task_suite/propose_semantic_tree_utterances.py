"""Propose richer event/recap realizations; never promote without validation.

The sampled symbolic world remains authoritative. This screen records
mechanical accepts and raw rejections so a stronger local LM can be compared
with the earlier small-model template pilot. Candidate forms are NOT an
approved grammar or training set.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SPECS = {
    'attributed_change': {
        'slots': ('[SOURCE]', '[OBJECT]', '[OLD]', '[NEW]'),
        'example': '[SOURCE] reported that the [OBJECT] changed from [OLD] to [NEW].',
        'fact': 'The object was OLD before the change, is NEW after it, and SOURCE reported the change.',
    },
    'color_recap': {
        'slots': ('[SOURCE]', '[OBJECT]', '[NEW]'),
        'example': 'For the [OBJECT], [SOURCE] recorded the final color as [NEW].',
        'fact': 'The final color of OBJECT is NEW. SOURCE reported this final state.',
    },
    'source_recap': {
        'slots': ('[OBJECT]', '[NEW]', '[SOURCE]'),
        'example': 'The [OBJECT] was [NEW] by the end, [SOURCE] reported.',
        'fact': 'The final color of OBJECT is NEW, and SOURCE was the person who reported its last change.',
    },
}
STYLES = ('short report', 'reference paragraph', 'newspaper summary',
          'inventory narrative', 'field report', 'plain-language update')
DEVICES = (
    'put the source attribution at the beginning',
    'put the source attribution at the end',
    'use a parenthetical attribution',
    'use a subordinate clause',
    'use a passive reporting construction',
    'use a fronted object phrase',
    'use two coordinated clauses in one sentence',
    'use a relative clause',
    'use a concise active clause',
    'vary the sentence opening without adding facts',
)
UNSUPPORTED = re.compile(
    r'\b(?:owner|owned|bought|sold|lost|found|broken|destroyed|stolen|'
    r'missing|painted|moved|stored|witness|police|court|yesterday|today|'
    r'tomorrow|morning|evening|first time|always|never|only|still|'
    r'not|no longer|decided|promised|expected|planned)\b', re.I)
ATTRIBUTION = re.compile(
    r'\b(?:report(?:ed|s)?|said|says|noted|stated|recorded|described|'
    r'confirmed|according to|account of|account from)\b', re.I)
CHANGE = re.compile(
    r'\b(?:changed|turned|shifted|became|went from|transitioned|'
    r'was revised|was updated)\b', re.I)


def prompt(kind, style, device):
    spec = SPECS[kind]
    system = ('Rewrite a symbolic fact as one varied, natural English '
              'sentence for a '+style+'. Keep every bracketed placeholder '
              'exactly once. State only the supplied fact. Output just the '
              'sentence, with no explanation or list marker.')
    user = (f"Meaning: {spec['fact']}\n"
            f"Basic form: {spec['example']}\n"
            f'For this variant, {device}. '
            'Use different syntax from the basic form while preserving the '
            'meaning and all placeholders exactly. Sentence:')
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': user}]


def filter_candidate(raw, kind):
    text = raw.strip().splitlines()[0].strip() if raw.strip() else ''
    text = re.sub(r'\[\s*(SOURCE|OBJECT|OLD|NEW)\s*\]',
                  lambda match: '['+match.group(1)+']', text)
    if not (text.endswith('.') and text.count('.') == 1 and
            18 <= len(text) <= 220):
        return None, 'sentence_shape'
    spec = SPECS[kind]
    slots = spec['slots']
    if any(text.count(slot) != 1 for slot in slots):
        return None, 'missing_or_repeated_slot'
    stripped = text
    for slot in slots:
        stripped = stripped.replace(slot, '')
    if '[' in stripped or ']' in stripped or re.search(r'\d', stripped):
        return None, 'extra_slot_or_number'
    if UNSUPPORTED.search(stripped):
        return None, 'unsupported_claim_word'
    if not ATTRIBUTION.search(stripped):
        return None, 'no_attribution'
    if kind == 'attributed_change':
        if not CHANGE.search(stripped):
            return None, 'no_change_relation'
        if not (text.index('[OBJECT]') < text.index('[NEW]') and
                text.index('[OLD]') < text.index('[NEW]')):
            return None, 'wrong_change_order'
    elif kind == 'color_recap':
        if not (text.index('[OBJECT]') < text.index('[NEW]') and
                text.index('[SOURCE]') < text.index('[NEW]')):
            return None, 'site_prefix_order'
    elif not (text.index('[OBJECT]') < text.index('[SOURCE]') and
              text.index('[NEW]') < text.index('[SOURCE]')):
        return None, 'site_prefix_order'
    return text, 'mechanically_accepted'


def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation='sdpa').to(args.device)
    model.eval()
    result = {'model': str(args.model), 'seed': args.seed,
              'n_per_kind': args.n_per_kind,
              'filter_scope': 'Slots, ordering, narrow relation lexemes, '
                              'and obvious unsupported words only. Mechanical '
                              'accepts still require independent semantic '
                              'inspection before any use in a DGP.',
              'kinds': {}}
    for kind in SPECS:
        reasons, proposals = Counter(), []
        for i in range(args.n_per_kind):
            messages = prompt(kind, STYLES[i % len(STYLES)],
                              DEVICES[i % len(DEVICES)])
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            inputs = tokenizer(input_text, return_tensors='pt').to(args.device)
            torch.manual_seed(args.seed + 100000*list(SPECS).index(kind) + i)
            with torch.no_grad():
                ids = model.generate(
                    **inputs, do_sample=True, temperature=args.temperature,
                    top_p=args.top_p, top_k=20,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(ids[0][inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True)
            candidate, reason = filter_candidate(raw, kind)
            reasons[reason] += 1
            proposals.append({'style': STYLES[i % len(STYLES)],
                              'device': DEVICES[i % len(DEVICES)],
                              'raw': raw, 'candidate': candidate,
                              'reason': reason})
        result['kinds'][kind] = {
            'counts': dict(reasons),
            'distinct_mechanical_candidates': len(set(
                row['candidate'] for row in proposals
                if row['candidate'] is not None)),
            'proposals': proposals}
        print(kind, result['kinds'][kind]['counts'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--n-per-kind', type=int, default=30)
    parser.add_argument('--seed', type=int, default=74251)
    parser.add_argument('--temperature', type=float, default=.7)
    parser.add_argument('--top-p', type=float, default=.8)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

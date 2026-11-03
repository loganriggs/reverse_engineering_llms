"""Propose typed sports clauses with a small local LM; retain candidates only.

The rejection filter checks slot mechanics and obvious unsupported claims.
Accepted forms still require semantic inspection before DGP adoption.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = ('/home/loganriggs/.cache/huggingface/hub/'
                 'models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/'
                 '989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
SLOTS = {
    'result': ('[WINNER]', '[LOSER]', '[WIN_SCORE]', '[LOSE_SCORE]'),
    'player': ('[PLAYER]', '[POINTS]'),
}
THEMES = ('sports report', 'newspaper recap', 'results roundup',
          'scoreboard note', 'basketball blog', 'wire-service story')
UNSUPPORTED = re.compile(
    r'\b(?:championship|playoff|season|streak|consecutive|overtime|'
    r'home|road|arena|stadium|quarter|half|night|yesterday|today|'
    r'injury|coach|shooting|rebounds|assists|record|favorite|underdog|'
    r'clinched|advanced|eliminated|ranked|first place|comeback)\b',
    re.IGNORECASE)
WIN_VERBS = re.compile(
    r'\b(?:beat|defeated|edged|topped|overcame|outscored|'
    r'prevailed over|triumphed over|won against|got past)\b', re.IGNORECASE)
POINT_VERBS = re.compile(
    r'\b(?:scored|posted|had|recorded|finished with|contributed|'
    r'put up|tallied|added)\b', re.IGNORECASE)


def prompt(kind, theme):
    if kind == 'result':
        example = '[WINNER] defeated [LOSER], [WIN_SCORE] to [LOSE_SCORE].'
        rule = ('The winner defeated the loser by the final score. '
                'Do not claim home/away, venue, streak, season, or anything '
                'else. Keep winning score before losing score.')
    else:
        example = '[PLAYER] scored [POINTS] points.'
        rule = ('The player scored that number of points. Do not add any '
                'other player, team, statistic, achievement, or game context.')
    system = ('You paraphrase one basketball fact into one natural '+theme+
              ' sentence. Output only the sentence. Keep each square-bracket '
              'placeholder exactly once. Do not replace them with examples.')
    user = 'Fact: '+example+'\nConstraint: '+rule+'\nParaphrase:'
    return [{'role':'system', 'content':system},
            {'role':'user', 'content':user}]


def filter_candidate(raw, kind):
    text = raw.strip().split('\n')[0].strip()
    slots = SLOTS[kind]
    if text.count('.') != 1 or not text.endswith('.') or len(text) > 180:
        return None, 'sentence_shape'
    if any(text.count(slot) != 1 for slot in slots):
        return None, 'missing_or_repeated_slot'
    stripped = text
    for slot in slots:
        stripped = stripped.replace(slot, '')
    if '[' in stripped or ']' in stripped or re.search(r'\d', stripped):
        return None, 'extra_slot_or_number'
    if UNSUPPORTED.search(stripped):
        return None, 'unsupported_claim_word'
    if kind == 'result':
        if not (text.index('[WINNER]') < text.index('[LOSER]') and
                text.index('[WIN_SCORE]') < text.index('[LOSE_SCORE]')):
            return None, 'slot_order'
        between = text[text.index('[WINNER]')+len('[WINNER]'):
                       text.index('[LOSER]')]
        if not WIN_VERBS.search(between) or re.search(
                r'\b(?:lost|loses|loss|fell|defeat at)\b', stripped,
                re.IGNORECASE):
            return None, 'result_semantics'
    elif not POINT_VERBS.search(text):
        return None, 'player_semantics'
    # Exclude untyped capitalized proper names beyond the starting word.
    words = re.findall(r'\b[A-Z][a-z]+\b', stripped)
    if len(words) > 1:
        return None, 'proper_name'
    return text, 'accepted_candidate'


def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, local_files_only=True).to(args.device)
    model.eval()
    output = {'model': args.model, 'seed': args.seed,
              'samples_per_kind': args.n_per_kind,
              'filter_note': ('Mechanical and lexical guard only; accepted '
                              'forms are unvalidated candidate templates.'),
              'kinds': {}}
    for kind in SLOTS:
        forms = []
        reasons = Counter()
        for i in range(args.n_per_kind):
            theme = THEMES[i % len(THEMES)]
            input_text = tokenizer.apply_chat_template(
                prompt(kind, theme), tokenize=False,
                add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors='pt').to(args.device)
            torch.manual_seed(args.seed + i + (100000 if kind == 'player' else 0))
            with torch.no_grad():
                ids = model.generate(
                    **inputs, do_sample=True, temperature=args.temperature,
                    top_p=args.top_p, max_new_tokens=80,
                    pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(ids[0][inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True)
            form, reason = filter_candidate(raw, kind)
            reasons[reason] += 1
            if form is not None:
                forms.append(form)
        output['kinds'][kind] = {
            'attempts': args.n_per_kind,
            'accepted_candidates': len(forms),
            'distinct_candidates': len(set(forms)),
            'rejection_reasons': dict(reasons),
            'forms': sorted(set(forms)),
        }
        print(kind, output['kinds'][kind]['accepted_candidates'],
              output['kinds'][kind]['distinct_candidates'],
              dict(reasons), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--seed', type=int, default=37051)
    p.add_argument('--n-per-kind', type=int, default=120)
    p.add_argument('--temperature', type=float, default=.9)
    p.add_argument('--top-p', type=float, default=.95)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()

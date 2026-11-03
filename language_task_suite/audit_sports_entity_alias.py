"""Audit natural full-name-to-surname opportunities and paired alias edits."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from tokenizers import Tokenizer

from audit_document_structure_vs_fineweb import structure
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_rotowire_document_source import game_key
from compare_v12_fineweb_fresh import first_words
from corpus_score_sentence_renderer import _sentences
from sports_document_dgp import game_partition, resources
from sports_entity_alias_dgp import (
    _render_tokens, candidates, nuisance_rename, realize, sample_plan,
)


def answer_token_index(document, tokenizer):
    encoded = tokenizer.encode(document.text, add_special_tokens=False)
    indices = [i for i, (start, end) in enumerate(encoded.offsets)
               if start <= document.answer_start < end]
    if len(indices) != 1 or encoded.offsets[indices[0]][1] != \
            document.answer_end or indices[0] == 0:
        return None
    return indices[0]


def natural_repeat_surfaces(row):
    """Conservative repeated-number surfaces, not an entailment parser."""
    sentences = list(_sentences(row['summary']))
    home, away = row['home_line']['TEAM-PTS'], row['vis_line']['TEAM-PTS']
    score_sentences = sum(any(
        sep in ('-', 'to', '–') and
        (left, right) in ((home, away), (away, home))
        for left, sep, right in zip(sentence, sentence[1:], sentence[2:]))
        for sentence in sentences)
    repeat_player = False
    box = row['box_score']
    for key, name in box['PLAYER_NAME'].items():
        parts = name.split()
        points = box['PTS'].get(key)
        if len(parts) != 2 or not str(points).isdigit():
            continue
        with_points = 0
        for sentence in sentences:
            if not any(sentence[i:i+2] == parts
                       for i in range(len(sentence)-1)):
                continue
            if points in sentence and any(word.lower() in ('point', 'points')
                                          for word in sentence):
                with_points += 1
        if with_points >= 2:
            repeat_player = True
            break
    return score_sentences >= 2, repeat_player


def audit(group, n, seed, tokenizer, alias_prior='roster',
          alias_exponent=0.):
    rows = [row for row in resources()[0]['rotowire'][0]
            if game_partition(game_key(row)) == group]
    inventory = Counter()
    gaps = []
    for row in rows:
        inventory['reports'] += 1
        options = candidates(row)
        inventory['reports_with_pair'] += bool(options)
        inventory['pairs'] += len(options)
        repeated_score, repeated_player = natural_repeat_surfaces(row)
        inventory['score_pair_in_two_sentences'] += repeated_score
        inventory['full_player_and_points_in_two_sentences'] += \
            repeated_player
        gaps.extend(target-source for _, _, source, target in options)
    rng = random.Random(seed)
    seen = set()
    counts = Counter()
    originals, edited, original_windows, edited_windows, manual = [], [], [], [], []
    hashes = []
    attempts = 0
    while len(seen) < n:
        attempts += 1
        if attempts > n*500:
            raise RuntimeError('Too few unique eligible games')
        plan = sample_plan(rng, group, alias_prior=alias_prior,
                           alias_exponent=alias_exponent)
        if plan.game_key in seen:
            continue
        seen.add(plan.game_key)
        hashes.append(hashlib.sha256(repr(plan.game_key).encode()).hexdigest())
        a, b = realize(plan, 'a'), realize(plan, 'b')
        assert a.text[a.source_start:a.source_end] == plan.first+' '+plan.alias_a
        assert b.text[b.source_start:b.source_end] == plan.first+' '+plan.alias_b
        assert a.text[a.answer_start:a.answer_end] == plan.alias_a
        assert b.text[b.answer_start:b.answer_end] == plan.alias_b
        strip_a = re.sub(r'\b'+re.escape(plan.alias_a)+r'\b',
                         '<ALIAS>', a.text)
        strip_b = re.sub(r'\b'+re.escape(plan.alias_b)+r'\b',
                         '<ALIAS>', b.text)
        assert strip_a == strip_b
        assert plan.original_surname not in a.text.split()
        assert plan.original_surname not in b.text.split()
        counts['paired_selective_substitution'] += 1
        nuisance = nuisance_rename(plan)
        if nuisance is not None:
            control = realize(plan, 'a', nuisance=True)
            assert control.answer == a.answer
            assert control.text[control.source_start:control.source_end] == \
                a.text[a.source_start:a.source_end]
            assert control.text[control.answer_start:control.answer_end] == \
                a.answer
            old, new = nuisance
            assert old != plan.original_surname and new != plan.alias_a
            assert re.sub(r'\b'+re.escape(new)+r'\b', '<OTHER>',
                          control.text) == re.sub(
                              r'\b'+re.escape(old)+r'\b', '<OTHER>',
                              a.text)
            counts['nuisance_preserves_target'] += 1
        for arm, doc in (('a', a), ('b', b)):
            index = answer_token_index(doc, tokenizer)
            counts[f'single_bpe_{arm}'] += index is not None
            for context in (320, 384, 512):
                counts[f'within_{context}_{arm}'] += \
                    index is not None and index < context
        originals.append(first_words(' '.join(plan.source_tokens), 100))
        edited.append(first_words(a.text, 100))
        original_text, spans = _render_tokens(plan.source_tokens)
        original_start = spans[plan.target_token][0]
        for text, index, collection in (
                (original_text, original_start, original_windows),
                (a.text, a.answer_start, edited_windows)):
            collection.append(' '.join(text[:index].split()[-25:]+
                                       text[index:].split()[:16]))
        if len(manual) < 8:
            before = ' '.join(plan.source_tokens[max(0, plan.source_token-12):
                                                 min(len(plan.source_tokens),
                                                     plan.target_token+12)])
            after = a.text[max(0, a.source_start-80):a.answer_end+80]
            manual.append({'original': before, 'alias': after,
                           'source_to_target_tokens': plan.target_token-
                                                      plan.source_token})
    return {'inventory': dict(inventory),
            'source_to_target_gap': {'min': min(gaps),
                                     'mean': sum(gaps)/len(gaps),
                                     'max': max(gaps)},
            'panel': {'unique_games': n, 'draw_attempts': attempts,
                      'game_sha256': hashes, 'checks': dict(counts),
                      'original_first100': structure(originals),
                      'alias_first100': structure(edited),
                      'original_target_windows': structure(original_windows),
                      'alias_target_windows': structure(edited_windows)},
            'manual': manual}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--seed', type=int, default=97501)
    p.add_argument('--alias-prior', choices=('roster', 'eligible_target'),
                   default='roster')
    p.add_argument('--alias-exponent', type=float, default=0.)
    args = p.parse_args()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    result = {'config': {**vars(args), 'output': str(args.output)},
              'groups': {},
              'caveat': 'Later surname surface matches are plausible '
                        'coreference, not independently verified entailment. '
                        'Manual examples are not committed. Natural source '
                        'reports have no full next-token oracle.'}
    for group in ('train', 'heldout'):
        row = audit(group, args.n, args.seed+(group == 'heldout')*100,
                    tokenizer, args.alias_prior, args.alias_exponent)
        manual = row.pop('manual')
        result['groups'][group] = row
        print(group, row['inventory'], row['panel']['checks'], flush=True)
        print('manual', manual, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

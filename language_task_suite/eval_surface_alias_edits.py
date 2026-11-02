"""Oracle-scored alias-key edits on the joint lexical/induction language."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from audit_surface_alias_mixture import supported_prefix
from eval_surface_mixture_edits import predict
from softmax_attention_control import configure_attention
from surface_alias_mixture import (ALIASES, AliasMixtureLanguage,
                                   _alias_definition, _is_parenthetical,
                                   alias_state, render)
from surface_mixture_v2 import PEOPLE, _frame, _indices, _split


def build_pairs(gen, n, length, edit_selection='first'):
    if edit_selection not in ('first', 'balanced'):
        raise ValueError(edit_selection)
    ids, labels, _ = gen.sample_batch(n, length+1)
    pairs = []
    for stream_index, (row, labs) in enumerate(zip(ids, labels)):
        tokens = [gen.id2token[i] for i in row]
        for t in range(1, len(row)):
            if labs[t] != 'mixed_verb':
                continue
            before = tokens[:t]
            history, clause = _split(before)
            mapping = alias_state(history)
            canonical = gen._canonical_prefix(history, clause)
            _, clean_clause = _split(canonical)
            subject_pos = _indices(_frame(clean_clause))[0]
            subject = clause[subject_pos]
            if subject not in ALIASES:
                continue
            first_use = not any(subject in old for old in history
                                if not _is_parenthetical(old))
            index = sum(len(old) for old in history)+subject_pos
            support = gen.next_distribution(before[:index])
            q_before = gen.next_distribution(before)
            changes = [('same_referent', mapping[subject])]
            other_aliases = [(alias, person) for alias, person in mapping.items()
                             if alias != subject and person != mapping[subject]]
            if edit_selection == 'balanced' and other_aliases:
                shift = (stream_index+t) % len(other_aliases)
                other_aliases = other_aliases[shift:]+other_aliases[:shift]
            changes.extend(('different_referent', alias)
                           for alias, _ in other_aliases)
            for kind, replacement in changes:
                if support.get(replacement, 0) <= 0:
                    continue
                after = before.copy()
                after[index] = replacement
                if not supported_prefix(gen, after, index):
                    continue
                q_after = gen.next_distribution(after)
                if kind == 'same_referent':
                    assert q_after == q_before
                elif q_after == q_before:
                    continue
                pairs.append(dict(kind=kind, original=[gen.token2id[x] for x in before],
                                  edited=[gen.token2id[x] for x in after],
                                  old_alias=subject, replacement=replacement,
                                  first_post_definition_use=first_use,
                                  old_referent=mapping[subject],
                                  new_referent=mapping.get(replacement, replacement),
                                  oracle_before=q_before, oracle_after=q_after))
                if kind == 'different_referent':
                    break
            offset = 0
            for old in history:
                definition = _alias_definition(old)
                if definition is not None and definition[0] == subject:
                    definition_index = offset
                    break
                offset += len(old)
            else:
                raise AssertionError('Alias subject without a definition')
            definition_support = gen.next_distribution(before[:definition_index])
            referents = list(PEOPLE)
            if edit_selection == 'balanced':
                shift = (stream_index*17+t) % len(referents)
                referents = referents[shift:]+referents[:shift]
            for referent in referents:
                if (referent == mapping[subject]
                        or referent in (person for alias, person in mapping.items()
                                        if alias != subject)
                        or definition_support.get(referent, 0) <= 0):
                    continue
                after = before.copy()
                after[definition_index] = referent
                if not supported_prefix(gen, after, definition_index):
                    continue
                q_after = gen.next_distribution(after)
                if q_after == q_before:
                    continue
                pairs.append(dict(kind='definition_referent',
                                  original=[gen.token2id[x] for x in before],
                                  edited=[gen.token2id[x] for x in after],
                                  old_alias=subject, replacement=subject,
                                  first_post_definition_use=first_use,
                                  old_referent=mapping[subject],
                                  new_referent=referent,
                                  oracle_before=q_before, oracle_after=q_after))
                break
            # Change a different alias's earlier definition. This oracle-null
            # control distinguishes keyed lookup from reading any definition.
            for other_alias, other_person in mapping.items():
                if other_alias == subject:
                    continue
                offset = 0
                for old in history:
                    definition = _alias_definition(old)
                    if definition is not None and definition[0] == other_alias:
                        break
                    offset += len(old)
                else:
                    raise AssertionError('Defined alias missing its clause')
                other_support = gen.next_distribution(before[:offset])
                candidates = list(PEOPLE)
                if edit_selection == 'balanced':
                    shift = (stream_index*29+t) % len(candidates)
                    candidates = candidates[shift:]+candidates[:shift]
                for referent in candidates:
                    if (referent == other_person
                            or referent in mapping.values()
                            or other_support.get(referent, 0) <= 0):
                        continue
                    after = before.copy()
                    after[offset] = referent
                    if not supported_prefix(gen, after, offset):
                        continue
                    q_after = gen.next_distribution(after)
                    if q_after != q_before:
                        continue
                    pairs.append(dict(kind='irrelevant_definition',
                                      original=[gen.token2id[x] for x in before],
                                      edited=[gen.token2id[x] for x in after],
                                      old_alias=subject, replacement=other_alias,
                                      first_post_definition_use=first_use,
                                      old_referent=other_person,
                                      new_referent=referent,
                                      oracle_before=q_before,
                                      oracle_after=q_after))
                    break
                break
            if (gen.initial_alias_definitions >= 2 and len(history) >= 2
                    and _alias_definition(history[0]) is not None
                    and _alias_definition(history[1]) is not None
                    and len(history[0]) == len(history[1]) == 7):
                first_alias = _alias_definition(history[0])[0]
                second_alias = _alias_definition(history[1])[0]
                if subject in (first_alias, second_alias):
                    swapped_order = history[1]+history[0]+[
                        token for old in history[2:] for token in old]+clause
                    assert supported_prefix(gen, swapped_order, 0)
                    assert gen.next_distribution(swapped_order) == q_before
                    pairs.append(dict(kind='definition_order',
                                      original=[gen.token2id[x] for x in before],
                                      edited=[gen.token2id[x] for x in swapped_order],
                                      old_alias=subject, replacement=subject,
                                      first_post_definition_use=first_use,
                                      old_referent=mapping[subject],
                                      new_referent=mapping[subject],
                                      oracle_before=q_before,
                                      oracle_after=q_before))
                    swapped_labels = before.copy()
                    swapped_labels[4], swapped_labels[11] = (
                        swapped_labels[11], swapped_labels[4])
                    if supported_prefix(gen, swapped_labels, 4):
                        q_swapped = gen.next_distribution(swapped_labels)
                    else:
                        q_swapped = q_before
                    if q_swapped != q_before:
                        other = (second_alias if subject == first_alias
                                 else first_alias)
                        pairs.append(dict(kind='alias_label_swap',
                                          original=[gen.token2id[x] for x in before],
                                          edited=[gen.token2id[x]
                                                  for x in swapped_labels],
                                          old_alias=subject, replacement=other,
                                          first_post_definition_use=first_use,
                                          old_referent=mapping[subject],
                                          new_referent=mapping[other],
                                          oracle_before=q_before,
                                          oracle_after=q_swapped))
    return pairs


def score(pairs, before, after, gen):
    result = {}
    for kind in ('same_referent', 'different_referent', 'definition_referent',
                 'irrelevant_definition',
                 'definition_order', 'alias_label_swap',
                 'same_referent_first_use', 'different_referent_first_use',
                 'definition_referent_first_use',
                 'irrelevant_definition_first_use',
                 'definition_order_first_use', 'alias_label_swap_first_use'):
        first = kind.endswith('_first_use')
        base_kind = kind.removesuffix('_first_use')
        indices = [i for i, pair in enumerate(pairs)
                   if pair['kind'] == base_kind and
                   (not first or pair['first_post_definition_use'])]
        if not indices:
            result[kind] = dict(n=0)
            continue
        oracle = np.zeros((len(indices), len(gen.vocab)), dtype=np.float32)
        for j, i in enumerate(indices):
            for token, p in pairs[i]['oracle_after'].items():
                oracle[j, gen.token2id[token]] += p
            for token, p in pairs[i]['oracle_before'].items():
                oracle[j, gen.token2id[token]] -= p
        model = after[indices]-before[indices]
        result[kind] = dict(n=len(indices),
                            oracle_tv=float(np.abs(oracle).sum(-1).mean()/2),
                            model_tv=float(np.abs(model).sum(-1).mean()/2),
                            effect_l1_error=float(np.abs(model-oracle).sum(-1).mean()),
                            no_change_l1_error=float(np.abs(oracle).sum(-1).mean()),
                            direction_cosine=float(np.mean([
                                np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
                                if np.linalg.norm(a)>1e-8 and np.linalg.norm(b)>1e-8
                                else 0.0 for a, b in zip(oracle, model)])))
        if base_kind == 'different_referent':
            q0 = np.zeros_like(oracle)
            q1 = np.zeros_like(oracle)
            for j, i in enumerate(indices):
                for token, p in pairs[i]['oracle_before'].items():
                    q0[j, gen.token2id[token]] = p
                for token, p in pairs[i]['oracle_after'].items():
                    q1[j, gen.token2id[token]] = p
            p0, p1 = before[indices], after[indices]
            avg = (q0+q1)/2
            tv = lambda x, y: float(np.abs(x-y).sum(-1).mean()/2)
            result[kind]['absolute_tv'] = dict(
                correct=(tv(p0, q0)+tv(p1, q1))/2,
                opposite=(tv(p0, q1)+tv(p1, q0))/2,
                equal_mix=(tv(p0, avg)+tv(p1, avg))/2)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--length', type=int, default=127)
    p.add_argument('--seed', type=int, default=20261106)
    p.add_argument('--edit-selection', choices=('first', 'balanced'),
                   default='first')
    p.add_argument('--generator-kwargs', type=json.loads,
                   help='Optional JSON object overriding the checkpoint DGP rates')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads((a.model_dir/'results.json').read_text())
    generator_kwargs = (a.generator_kwargs if a.generator_kwargs is not None
                        else saved.get('generator_kwargs', {}))
    gen = AliasMixtureLanguage(seed=a.seed, **generator_kwargs)
    pairs = build_pairs(gen, a.n, a.length, a.edit_selection)
    assert sum(pair['kind'] == 'same_referent' for pair in pairs) > 50
    assert sum(pair['kind'] == 'different_referent' for pair in pairs) > 20
    record = next(row for row in saved['records'] if row['depth'] == a.depth)
    model = configure_attention(BilinearGPT(Config(**record['model_config'])),
                                saved['config'].get('attention', 'bilinear'))
    model.load_state_dict(torch.load(a.model_dir/f'model_depth{a.depth}.pt',
                                     map_location='cpu', weights_only=True))
    model.eval()
    p0 = predict(model, [pair['original'] for pair in pairs], len(gen.vocab))
    p1 = predict(model, [pair['edited'] for pair in pairs], len(gen.vocab))
    result = dict(config={**vars(a), 'generator_kwargs': generator_kwargs,
                          'model_dir': str(a.model_dir),
                          'output': str(a.output)},
                  counts=dict(Counter(pair['kind'] for pair in pairs)),
                  scores=score(pairs, p0, p1, gen),
                  examples=[dict(kind=pair['kind'],
                                 original=render(pair['original']),
                                 edited=render(pair['edited']),
                                 old_referent=pair['old_referent'],
                                 new_referent=pair['new_referent'])
                            for pair in pairs[:4]],
                  caveat='The model was trained on all alias-person pairings. '
                         'These are fresh stream/edit seeds, not whole-pair '
                         'training holdouts. Same-referent edits have exactly '
                         'zero oracle effect; different-referent edits have '
                         'supported full prefixes and a changed verb oracle.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'], result['scores'])


if __name__ == '__main__':
    main()

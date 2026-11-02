"""Supported Catalog interventions for keyed field-value retrieval."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from eval_surface_mixture_edits import predict, score
from multi_genre_v2_language import (CATALOG_FIELDS, CATALOG_OBJECTS,
                                     CATALOG_PEOPLE,
                                     CATALOG_STATUSES, MultiGenreV2Language,
                                     catalog_state)


def query_metadata(words):
    records, asked, clause = catalog_state(words[2:])
    form = clause[0]
    if form == 'Who':
        field, object_at = 'owner', 3
    elif form == 'What':
        field, object_at = 'status', 6
    else:
        field, object_at = ('owner' if clause[4] == 'who' else 'status'), 2
    start = max(i for i, word in enumerate(words)
                if word in ('Who', 'What', 'For'))
    assert words[start+object_at] == clause[object_at]
    return records, asked, field, clause[object_at], start+object_at


def record_value_indices(words):
    result = {}
    for i, word in enumerate(words):
        if word == 'The' and i+2 < len(words) and words[i+1] in CATALOG_OBJECTS:
            obj = words[i+1]
            if words[i+2] == 'belongs':
                result[obj] = {'owner': i+4, 'status': i+7}
            elif words[i+2] == 'is':
                result[obj] = {'status': i+3, 'owner': i+7}
    return result


def supported(gen, words, changed_at):
    for at in range(changed_at, len(words)):
        q, _ = gen.next_distribution(words[:at])
        if q.get(words[at], 0.) <= 0:
            return False
    return True


def catalog_edits(gen, prefix, rotate=0):
    words = [gen.id2token[int(i)] for i in prefix]
    assert words[0] == 'Catalog'
    q0, label = gen.next_distribution(words)
    assert label in ('catalog_owner_answer', 'catalog_status_answer')
    records, asked, field, obj, key_at = query_metadata(words)
    indices = record_value_indices(words)
    assert set(records) == set(indices)
    results = []
    specifications = [('catalog_query_key', key_at,
                       [other for other in records if other != obj
                        and records[other][field] != records[obj][field]], True)]
    other_field = next(f for f in CATALOG_FIELDS if f != field)
    specifications.append(('catalog_relevant_value', indices[obj][field],
                           list(CATALOG_PEOPLE if field == 'owner'
                                else CATALOG_STATUSES), True))
    specifications.append(('catalog_other_field', indices[obj][other_field],
                           list(CATALOG_PEOPLE if other_field == 'owner'
                                else CATALOG_STATUSES), False))
    other_obj = next((other for other in records if other != obj), None)
    if other_obj is not None:
        specifications.append(('catalog_irrelevant_record',
                               indices[other_obj][field],
                               list(CATALOG_PEOPLE if field == 'owner'
                                    else CATALOG_STATUSES), False))
    for kind, at, options, should_change in specifications:
        if not options:
            continue
        offset = rotate % len(options)
        for word in options[offset:]+options[:offset]:
            if word == words[at]:
                continue
            edited = list(words)
            edited[at] = word
            if not supported(gen, edited, at):
                continue
            q1, edited_label = gen.next_distribution(edited)
            assert edited_label == label
            if (q1 != q0) != should_change:
                continue
            results.append(dict(kind=kind, field=field,
                                first_query=(len(asked) == 0),
                                original=list(prefix),
                                edited=[gen.token2id[w] for w in edited],
                                q_before=q0, q_after=q1))
            break
    return results


def build_pairs(gen, nstreams, length):
    rows, labels, _ = gen.sample_batch(nstreams, length+1)
    pairs = []
    for stream, (row, labs) in enumerate(zip(rows, labels)):
        if gen.id2token[row[0]] != 'Catalog':
            continue
        for at, label in enumerate(labs):
            if label in ('catalog_owner_answer', 'catalog_status_answer'):
                pairs.extend(catalog_edits(gen, row[:at], stream+at))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--depths', default='2,3')
    p.add_argument('--nstreams', type=int, default=1024)
    p.add_argument('--length', type=int, default=127)
    p.add_argument('--seed', type=int, default=70011)
    p.add_argument('--record-rate', type=float, default=.60)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    from softmax_attention_control import configure_attention

    gen = MultiGenreV2Language(seed=a.seed, record_rate=a.record_rate)
    pairs = build_pairs(gen, a.nstreams, a.length)
    originals = [p['original'] for p in pairs]
    edits = [p['edited'] for p in pairs]
    result = dict(config={**vars(a), 'model_dir': str(a.model_dir),
                          'output': str(a.output)},
                  pair_counts=dict(Counter(p['kind'] for p in pairs)),
                  first_pair_counts=dict(Counter(p['kind'] for p in pairs
                                                if p['first_query'])),
                  scores={}, first_query_scores={})
    records = json.loads((a.model_dir/'results.json').read_text())['records']
    training_config = json.loads((a.model_dir/'results.json').read_text())['config']
    for depth in [int(s) for s in a.depths.split(',')]:
        record = next(r for r in records if r['depth'] == depth)
        model = configure_attention(BilinearGPT(Config(**record['model_config'])),
                                    training_config.get('attention', 'bilinear'))
        model.load_state_dict(torch.load(a.model_dir/f'model_depth{depth}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        before = predict(model, originals, len(gen.vocab))
        after = predict(model, edits, len(gen.vocab))
        result['scores'][str(depth)] = score(pairs, before, after, gen)
        first_indices = [i for i, pair in enumerate(pairs)
                         if pair['first_query']]
        result['first_query_scores'][str(depth)] = score(
            [pairs[i] for i in first_indices], before[first_indices],
            after[first_indices], gen)
        print(depth, {kind: round(row['effect_l1_error'], 3)
                      for kind, row in result['scores'][str(depth)].items()},
              flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

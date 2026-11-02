"""Bayes query-key-blind answer baseline for the Catalog test stream."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from multi_genre_v2_language import MultiGenreV2Language, catalog_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=9101)
    p.add_argument('--n', type=int, default=512)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--record-rate', type=float, default=.60)
    p.add_argument('--genre-weights', default='',
                   help='Optional five comma-separated genre weights')
    p.add_argument('--output', type=Path,
                   default=Path('language_task_suite/multi_genre_v2_key_blind.json'))
    a = p.parse_args()
    weights = (tuple(float(x) for x in a.genre_weights.split(','))
               if a.genre_weights else None)
    kwargs = {'record_rate': a.record_rate}
    if weights is not None:
        kwargs['genre_weights'] = weights
    gen = MultiGenreV2Language(seed=a.seed, **kwargs)
    rows, labels, _ = gen.sample_batch(a.n, a.length)
    losses = defaultdict(list)
    first_losses = defaultdict(list)
    for row, labs in zip(rows, labels):
        if gen.id2token[row[0]] != 'Catalog':
            continue
        for at, label in enumerate(labs):
            if label not in ('catalog_owner_answer', 'catalog_status_answer'):
                continue
            words = [gen.id2token[i] for i in row[:at]]
            records, asked, _ = catalog_state(words[2:])
            field = 'owner' if label == 'catalog_owner_answer' else 'status'
            eligible = gen._choices(records, asked, field)
            # Distinct values make the uniform marginal over eligible answers
            # the exact Bayes predictor when the selected object is hidden.
            assert len({records[obj][field] for obj in eligible}) == len(eligible)
            loss = math.log(len(eligible))
            losses[label].append(loss)
            if not asked:
                first_losses[label].append(loss)
    result = dict(config={**vars(a), 'output': str(a.output)},
                  query_blind_kl={name: dict(n=len(values),
                                             kl=sum(values)/len(values))
                                  for name, values in losses.items()},
                  first_query_blind_kl={name: dict(n=len(values),
                                                   kl=sum(values)/len(values))
                                        for name, values in first_losses.items()},
                  caveat='The baseline sees the field, record values, and '
                         'previously queried object-field pairs, but not the '
                         'current query object. It is a capability control, '
                         'not an interpretation of any model.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['query_blind_kl'])


if __name__ == '__main__':
    main()

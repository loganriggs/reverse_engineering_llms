"""Calibrate the sampled alias-frequency exponent to eligible natural names."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

from sports_entity_alias_dgp import sample_plan, single_bpe_surname_pool


def summary(counts):
    total = sum(counts.values())
    return {'samples': total, 'types': len(counts),
            'top10_share': sum(v for _, v in counts.most_common(10))/total,
            'entropy_nats': -sum(v/total*math.log(v/total)
                                 for v in counts.values())}


def js_divergence(a, b, support):
    na, nb = sum(a.values()), sum(b.values())
    pa = {word: a[word]/na for word in support}
    pb = {word: b[word]/nb for word in support}
    midpoint = {word: (pa[word]+pb[word])/2 for word in support}
    return .5*sum(p*math.log(p/midpoint[word]) for word, p in pa.items()
                    if p) + .5*sum(p*math.log(p/midpoint[word])
                                    for word, p in pb.items() if p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=97801)
    p.add_argument('--n', type=int, default=2000)
    args = p.parse_args()
    exponents = (0., .5, 1., 1.25, 1.5, 2.)
    support = set(single_bpe_surname_pool())
    natural = Counter()
    # Use one world-draw stream for the reference; each alias arm has its own
    # deterministic stream. No model fitting or held-out reports enter here.
    rng = random.Random(args.seed)
    for _ in range(args.n):
        natural[sample_plan(rng, 'train').original_surname] += 1
    natural_restricted = Counter({word: count for word, count in natural.items()
                                  if word in support})
    arms = {}
    for prior in ('roster', 'eligible_target'):
        for exponent in exponents:
            rng = random.Random(args.seed)
            aliases = Counter()
            for _ in range(args.n):
                aliases[sample_plan(rng, 'train', alias_prior=prior,
                                    alias_exponent=exponent).alias_a] += 1
            arms[f'{prior}:{exponent}'] = {
                'frequency': summary(aliases),
                'js_to_eligible_natural_one_bpe':
                    js_divergence(natural_restricted, aliases, support)}
    result = {'config': {**vars(args), 'output': str(args.output)},
              'one_bpe_pool_size': len(support),
              'eligible_natural_all': summary(natural),
              'eligible_natural_one_bpe': summary(natural_restricted),
              'eligible_natural_one_bpe_fraction':
                  sum(natural_restricted.values())/args.n,
              'arms': arms,
              'caveat': 'Matches only the eligible RotoWire target-surname '
                        'marginal, restricted to one-BPE surnames. It does '
                        'not certify Zipf behavior or natural joint name, '
                        'discourse, and task distributions.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['eligible_natural_one_bpe_fraction'])
    for exponent, row in arms.items():
        print(exponent, row['frequency']['top10_share'],
              row['js_to_eligible_natural_one_bpe'])


if __name__ == '__main__':
    main()

"""Check the specified marginal and joint rule-gate law by repeated draws."""

import argparse
import hashlib
import json
from pathlib import Path
import random

from corpus_rule_mixture_dgp import Config, draw, eligibility


def audit(preview: Path, draws: int, seed: int, p: float, c: float):
    rows = json.loads(preview.read_text())
    source = next(row for row in rows if {'bracket', 'literal_copy'} <=
                  eligibility(row['preview']))
    result = {}
    for name, strength in (('independent', 0.), ('coupled', c)):
        counts = {'bracket': 0, 'literal_copy': 0, 'joint': 0}
        for i in range(draws):
            item = draw(source, random.Random(seed+i),
                        Config(p, strength))
            active = set(item.document.applied)
            a, b = 'bracket' in active, 'literal_copy' in active
            counts['bracket'] += a
            counts['literal_copy'] += b
            counts['joint'] += a and b
        observed = {key: value/draws for key, value in counts.items()}
        expected = {'bracket': p, 'literal_copy': p,
                    'joint': p*p+strength*strength}
        assert all(abs(observed[key]-expected[key]) < .03
                   for key in expected), (name, observed, expected)
        result[name] = {'observed': observed, 'expected': expected}
    return {'config': {'draws': draws, 'seed': seed, 'gate_rate': p,
                       'coupling_strength': c,
                       'preview_sha256': hashlib.sha256(
                           preview.read_bytes()).hexdigest(),
                       'eligible_source_sha256': source['sha256']},
            'results': result,
            'scope': 'Repeated draws of one source eligible for bracket '
                     'and literal copy. Empirical activation is checked '
                     'against the explicit gate law, not a population '
                     'estimate of natural rule frequencies.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preview', type=Path, required=True)
    p.add_argument('--draws', type=int, default=5000)
    p.add_argument('--seed', type=int, default=410000)
    p.add_argument('--gate-rate', type=float, default=.5)
    p.add_argument('--coupling-strength', type=float, default=.4)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.preview, a.draws, a.seed,
                   a.gate_rate, a.coupling_strength)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['results'], flush=True)


if __name__ == '__main__':
    main()

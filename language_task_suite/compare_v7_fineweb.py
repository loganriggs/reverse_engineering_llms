"""Compare v7 form statistics at diagnostic and low task rates to FineWeb."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_surface_repetition import form_cues, score
from surface_mixture_v7 import SurfaceMixtureV7, render
from surface_mixture_v8 import SurfaceMixtureV8


HERE = Path(__file__).resolve().parent
ROWS = (HERE.parents[1] / 'reverse_engineer/tensor_language/'
        'basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt')


def run(rows_path=ROWS, length=128, seed=12001, generator='v7'):
    rows = torch.load(rows_path, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    natural = [tokenizer.decode(row[:length].tolist(), skip_special_tokens=False)
               for row in rows]
    n = len(natural)
    configurations = {
        'diagnostic': dict(section_rate=.18 if generator == 'v7' else .40,
                           detail_rate=.80,
                           scope_question_rate=.40 if generator == 'v7' else .50,
                           quote_rate=.08,
                           source_rate=.15),
        'low_rate': dict(section_rate=.06, detail_rate=.80,
                         scope_question_rate=.08, quote_rate=.08,
                         source_rate=.15),
    }
    output = dict(source_rows=str(rows_path), generator=generator,
                  source_sha256=hashlib.sha256(rows.numpy().tobytes()).hexdigest(),
                  n=n, length=length, seed=seed,
                  fineweb=dict(forms=form_cues(natural),
                               repetition=score(natural, 40)))
    for name, kwargs in configurations.items():
        gen_class = {'v7': SurfaceMixtureV7, 'v8': SurfaceMixtureV8}[generator]
        gen = gen_class(seed=seed, counterbalanced_queries=True, **kwargs)
        texts = []
        labels = Counter()
        for _ in range(n):
            ids, labs, _ = gen.sample_sequence(length)
            texts.append(render(ids))
            labels.update(labs)
        output[name] = dict(generator_kwargs=kwargs,
                            forms=form_cues(texts),
                            repetition=score(texts, 40),
                            labels=dict(labels))
    output['caveat'] = ('Surface comparison only, with different tokenizers; '
                        'low_rate is not fitted to natural dependency prevalence. '
                        'Cue rates and answer-site counts are not interchangeable.')
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=Path, default=ROWS)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--seed', type=int, default=12001)
    p.add_argument('--generator', choices=('v7', 'v8'), default='v7')
    p.add_argument('--output', type=Path,
                   default=HERE/'surface_mixture_v7_fineweb_comparison.json')
    a = p.parse_args()
    result = run(a.rows, a.length, a.seed, a.generator)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for name in ('fineweb', 'diagnostic', 'low_rate'):
        print(name, result[name]['forms'],
              result[name].get('labels', {}).get('scope_answer', 0))

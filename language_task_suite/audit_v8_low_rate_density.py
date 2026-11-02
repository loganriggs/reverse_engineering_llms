"""Count rare diagnostic sites under the v8 lower-rate surface probe."""

import json
from collections import Counter
from pathlib import Path

from surface_mixture_v8 import SurfaceMixtureV8


HERE = Path(__file__).resolve().parent
KWARGS = dict(section_rate=.06, detail_rate=.80, scope_question_rate=.08,
              quote_rate=.08, source_rate=.15,
              counterbalanced_queries=True)


def run(seed=12691):
    panels = []
    for n, length in ((2000, 128), (1000, 256)):
        gen = SurfaceMixtureV8(seed=seed, **KWARGS)
        labels = Counter()
        for _ in range(n):
            _, row_labels, _ = gen.sample_sequence(length)
            labels.update(row_labels)
        panels.append(dict(n=n, length=length, labels=dict(labels)))
    return dict(seed=seed, generator='v8', generator_kwargs=KWARGS,
                panels=panels,
                caveat='Arbitrary lower task rates for a surface probe; not '
                       'a natural dependency prevalence estimate.')


if __name__ == '__main__':
    result = run()
    (HERE/'surface_mixture_v8_low_rate_density.json').write_text(
        json.dumps(result, indent=2)+'\n')
    print([(row['n'], row['length'], row['labels'].get('scope_answer', 0))
           for row in result['panels']])

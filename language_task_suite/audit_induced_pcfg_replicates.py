"""Synthetic-seed variability for the corpus-PCFG/FineWeb form screen."""

import argparse
import json
import statistics
from pathlib import Path

from audit_induced_pcfg_form import audit


METRICS = ('word_types', 'word_singletons', 'top10_word_share',
           'adjacent_pair_types', 'top10_adjacent_pair_share')


def replicate(grammar_base, preview_path, manifest_path,
              first_seed=58001, n=10):
    rows = [audit(grammar_base, preview_path, manifest_path,
                  first_seed+i) for i in range(n)]
    natural = {k: rows[0]['natural'][k] for k in METRICS}
    assert all(all(row['natural'][k] == natural[k] for k in METRICS)
               for row in rows)
    return dict(grammar_base_hashes={
                    'grammar': rows[0]['input_grammar_sha256'],
                    'lexicon': rows[0]['input_lexicon_sha256']},
                first_seed=first_seed, n=n, natural=natural,
                generated={k: dict(mean=statistics.mean(
                                      row['generated'][k] for row in rows),
                                   sd=statistics.stdev(
                                      row['generated'][k] for row in rows),
                                   minimum=min(row['generated'][k]
                                               for row in rows),
                                   maximum=max(row['generated'][k]
                                               for row in rows))
                           for k in METRICS},
                sample_failures=[row['sample_failures'] for row in rows],
                caveat='Synthetic seed variation on one fixed 43-document '
                       'FineWeb reference panel; not a confidence interval '
                       'for all web text or all grammar-induction samples.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--grammar-base', type=Path, required=True)
    p.add_argument('--preview-json', type=Path, required=True)
    p.add_argument('--manifest-json', type=Path,
                   default=Path(__file__).parent/'results'/
                   'fineweb_long_panel_25000_n48_manifest.json')
    p.add_argument('--first-seed', type=int, default=58001)
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = replicate(a.grammar_base, a.preview_json,
                       a.manifest_json, a.first_seed, a.n)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['generated'])

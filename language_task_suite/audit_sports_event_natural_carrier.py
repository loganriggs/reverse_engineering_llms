"""Paired natural-carrier form control for event-grounded run claims."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from audit_document_structure_vs_fineweb import structure
from audit_sports_report_run_mentions import RUN
from compare_v12_fineweb_fresh import first_words
from sports_document_dgp import _detokenize
from sports_event_natural_carrier import eligible_event_carriers, realize
from sports_event_world_renderer import run_values_from_visible_prefix


def window(text, start):
    return ' '.join(text[:start].split()[-25:]+text[start:].split()[:16])


def human_run_windows(source, group, carrier_words):
    out = {}
    for key, row, _ in eligible_event_carriers(source, group,
                                                carrier_words):
        text = _detokenize(row['summary'])
        match = RUN.search(text)
        if match:
            out.setdefault(key, window(text, match.start(1)))
    return tuple(out.values())


def panel(seed, n, words, source, group, carrier_words):
    available = eligible_event_carriers(source, group, carrier_words)
    rng = random.Random(seed)
    seen = set()
    natural, inserted, rule_windows = [], [], []
    counts = Counter()
    while len(inserted) < n:
        counts['draw_attempts'] += 1
        if counts['draw_attempts'] > 100*n:
            raise RuntimeError('Too few unique eligible carrier games')
        doc = realize(rng, source, group, carrier_words)
        if doc.game_id in seen:
            continue
        seen.add(doc.game_id)
        for site in doc.sites:
            assert doc.text[site.start:site.end] == site.answer
            values = run_values_from_visible_prefix(doc.text[:site.start])
            assert site.answer == str(values[0] if site.kind == 'run_for'
                                      else values[1])
            counts['visible_run_sites_checked'] += 1
        original_crop = first_words(doc.original, words)
        inserted_crop = first_words(doc.text, words)
        if min(len(original_crop.split()),
               len(inserted_crop.split())) < words:
            continue
        natural.append(original_crop)
        inserted.append(inserted_crop)
        rule_windows.append(window(doc.text, doc.sites[0].start))
    human_windows = human_run_windows(source, group, carrier_words)
    if len(human_windows) < 10:
        raise RuntimeError('Too few human run windows for local control')
    local_n = min(10, n)
    local = []
    for i in range(10):
        local_rng = random.Random(seed+30000+i)
        human = local_rng.sample(list(human_windows), local_n)
        generated = local_rng.sample(rule_windows, local_n)
        metric = 'repeated_across_docs_trigram_occurrence_share'
        local.append({'generated': structure(generated)[metric],
                      'human_run': structure(human)[metric]})
    return {'eligible_rows': len(available),
            'counts': dict(counts),
            'paired_original': structure(natural),
            'with_run_insertion': structure(inserted),
            'human_run_window_pool': len(human_windows),
            'local_window_n': local_n,
            'local_repetitions': local}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10305)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--words', type=int, default=100)
    parser.add_argument('--carrier-words', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = {'config': vars(args) | {'output': str(args.output)},
           'arms': {},
           'caveat': 'Empirical human report remains as carrier, and only '
                     'the inserted run block has generated semantics. '
                     'Local-window control uses ten resamples of ten '
                     'distinct reports and is noisy for small pools.'}
    for source in ('rotowire', 'sbnation'):
        for group in ('train', 'heldout'):
            key = source+'_'+group
            result = panel(args.seed+100*len(out['arms']), args.n,
                           args.words, source, group, args.carrier_words)
            out['arms'][key] = result
            metric = 'repeated_across_docs_trigram_occurrence_share'
            print(key, result['eligible_rows'],
                  result['paired_original'][metric],
                  result['with_run_insertion'][metric],
                  result['human_run_window_pool'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()

"""Rule integrity and form audit for variable typed document graphs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from audit_document_structure_vs_fineweb import structure
from audit_document_world_v5_clause_realizations import _edited_world
from audit_surface_repetition import WORD
from compare_v12_fineweb_fresh import first_words
from document_world_v10_fact_acts import (
    answer_from_visible_prefix, realize as realize_v10)
from document_world_v14_variable_graph import draw_graph


ARMS = {
    'fixed_3': dict(n_objects=(3, 3), extra_events=(2, 5),
                    revisit_rate=.55),
    'variable': dict(n_objects=(2, 6), extra_events=(2, 10),
                     revisit_rate=.55),
    'variable_spread': dict(n_objects=(2, 6), extra_events=(2, 10),
                            revisit_rate=.1),
    'variable_revisit': dict(n_objects=(2, 6), extra_events=(2, 10),
                             revisit_rate=.9),
}


def check(doc):
    for site in doc.sites:
        assert doc.tokens[site.index] == site.answer
        assert answer_from_visible_prefix(
            doc.tokens[:site.index], lexical_enabled=True) == site.answer


def site_window(doc, site, width):
    text = doc.text()
    words = list(WORD.finditer(text))
    if len(words) < width:
        return None
    target = len(WORD.findall(' '.join(doc.tokens[:site.index])))
    begin = max(0, min(target-width//2, len(words)-width))
    return text[words[begin].start():words[begin+width-1].end()]


def arm(seed, n, words, width, kwargs):
    rng = random.Random(seed)
    counts = Counter()
    firsts = []
    windows = defaultdict(list)
    examples = []
    while len(firsts) < n:
        world = draw_graph(rng, source_rate=.5, act_rate=.8, **kwargs)
        doc = realize_v10(world, lexical_enabled=True,
                          syntax_entropy=1., aside_rate=.3,
                          context_rate=.35)
        counts['draws'] += 1
        check(doc)
        counts['all_sites'] += len(doc.sites)
        crop = first_words(doc.text(), words)
        if len(WORD.findall(crop)) < words:
            continue
        counts['eligible'] += 1
        counts['objects/'+str(len(world['initial']))] += 1
        counts['events/'+str(len(world['events']))] += 1
        previously_touched = {world['events'][0]['object']}
        for event in world['events'][1:-1]:
            counts['extra_events'] += 1
            counts['revisited_extra_events'] += (
                event['object'] in previously_touched)
            previously_touched.add(event['object'])
        counts.update('site/'+site.family for site in doc.sites)
        firsts.append(crop)
        for family in set(site.family for site in doc.sites):
            site = next(s for s in doc.sites if s.family == family)
            window = site_window(doc, site, width)
            if window:
                windows[family].append(window)
        altered_world = _edited_world(world, world['recap_plan'][0])
        changed = realize_v10(altered_world, lexical_enabled=True,
                              syntax_entropy=1., aside_rate=.3,
                              context_rate=.35)
        check(changed)
        assert changed.sites[0].answer != doc.sites[0].answer
        counts['supported_edit_checks'] += 1
        if len(examples) < 2:
            examples.append(doc.text()[:700])
        assert counts['draws'] <= 100*n
    return {'counts': dict(counts), 'first100_form': structure(firsts),
            'site41_counts': {k: len(v) for k, v in windows.items()},
            'site41_form': {k: structure(v) for k, v in windows.items()},
            'examples': examples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--window-words', type=int, default=41)
    p.add_argument('--seed', type=int, default=54101)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    panels = {f'seed_{i}': {name: arm(
        a.seed+10000*i+100*j, a.n, a.words,
        a.window_words, kwargs)
        for j, (name, kwargs) in enumerate(ARMS.items())}
        for i in range(2)}
    result = {'config': {'n': a.n, 'words': a.words,
                         'window_words': a.window_words,
                         'seeds': [a.seed, a.seed+10000],
                         'arms': ARMS},
              'panels': panels,
              'caveat': 'Each arm samples independent worlds; first-100-word '
                        'form conditions on documents long enough in every '
                        'arm. Exact all-token next-token oracle unavailable. '
                        'Visible-answer and supported-edit checks cover '
                        'scored sites. No natural source text exported.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for seed, rows in panels.items():
        for name, row in rows.items():
            f = row['first100_form']
            print(seed, name, 'draws', row['counts']['draws'],
                  'types', f['word_types'], 'top10',
                  round(f['top10_word_share'], 3), 'trigrams',
                  round(f['repeated_across_docs_trigram_occurrence_share'], 3),
                  flush=True)


if __name__ == '__main__':
    main()

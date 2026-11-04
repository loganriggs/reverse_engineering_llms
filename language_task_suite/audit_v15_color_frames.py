"""Paired semantic/edit and surface-form audit of two new color frames."""

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
    answer_from_visible_prefix as answer_v10, realize as realize_v10)
from document_world_v14_variable_graph import draw_graph
from document_world_v15_color_frames import (
    answer_from_visible_prefix as answer_v15, event_plan,
    realize as realize_v15)


def check(doc, parser):
    for site in doc.sites:
        assert doc.tokens[site.index] == site.answer
        assert parser(doc.tokens[:site.index], lexical_enabled=True) == site.answer


def audit(n, seed, words):
    rng = random.Random(seed)
    texts = defaultdict(list)
    counts = Counter()
    examples = []
    for _ in range(n):
        world = draw_graph(rng)
        base = realize_v10(world, lexical_enabled=True,
                           syntax_entropy=1., aside_rate=.3,
                           context_rate=.35)
        varied = realize_v15(world, lexical_enabled=True,
                             syntax_entropy=1., aside_rate=.3,
                             context_rate=.35)
        check(base, answer_v10)
        check(varied, answer_v15)
        assert [(s.family, s.answer) for s in base.sites] == [
            (s.family, s.answer) for s in varied.sites]
        plan = event_plan(world)
        counts['color_events'] += sum(e['attribute'] == 'color'
                                      for e in world['events'])
        counts['new_color_frame_source_first'] += sum(
            sentence[1:2] == ['described'] for sentence in plan)
        counts['new_color_frame_action_first'] += sum(
            sentence[:1] == ['After'] for sentence in plan)
        counts.update('site/'+s.family for s in varied.sites)
        for name, doc in (('base', base), ('v15', varied)):
            crop = first_words(doc.text(), words)
            if len(WORD.findall(crop)) >= words:
                texts[name].append(crop)
        changed_world = _edited_world(world, world['recap_plan'][0])
        changed = realize_v15(changed_world, lexical_enabled=True,
                              syntax_entropy=1., aside_rate=.3,
                              context_rate=.35)
        check(changed, answer_v15)
        assert changed.sites[0].answer != varied.sites[0].answer
        counts['supported_edit_checks'] += 1
        if len(examples) < 2:
            examples.append(varied.text()[:700])
    return {'counts': dict(counts),
            'form': {name: structure(rows) for name, rows in texts.items()},
            'form_panel_counts': {name: len(rows) for name, rows in texts.items()},
            'examples': examples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--seed', type=int, default=56601)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    panels = {str(a.seed+i): audit(a.n, a.seed+i, a.words)
              for i in range(2)}
    report = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()},
              'panels': panels,
              'scope': 'Paired worlds and independent visible-prefix answer '
                       'checks at all scored sites, plus one supported edit '
                       'per world. No all-token oracle or natural-language '
                       'equivalence claim. Form is first-100-word screen.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    for seed, row in panels.items():
        print(seed, row['counts'], flush=True)
        for name, form in row['form'].items():
            print(name, row['form_panel_counts'][name],
                  'types', form['word_types'], 'top10',
                  round(form['top10_word_share'], 3),
                  'trigram', round(form[
                      'repeated_across_docs_trigram_occurrence_share'], 3),
                  flush=True)


if __name__ == '__main__':
    main()

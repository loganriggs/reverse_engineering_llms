"""Find which sampled rule-site families dominate the language-form deficit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from audit_document_structure_vs_fineweb import structure
from audit_rule_window_form import word_window
from audit_semantic_syntax_opportunities import WORD, first_words
from document_proposition_plan import sample_plan
from document_proposition_surface import answer_from_visible_prefix, realize


def window_for_site(document, site, width):
    target = len(WORD.findall(' '.join(document.tokens[:site.index])))
    return word_window(document.text(), target, width)


def is_question_site(document, site):
    prefix = document.tokens[:site.index]
    start = max((i+1 for i, token in enumerate(prefix) if token == '.'),
                default=0)
    return 'Answer' in prefix[start:]


def audit(preview, n, words, width, seed, surface_version=3):
    rows = json.loads(preview.read_text())
    if len(rows) < n+1:
        raise ValueError('Need separate natural reference rows')
    rng = random.Random(seed)
    site_rng = random.Random(seed ^ 0xADD72)
    windows = defaultdict(list)
    question_windows = defaultdict(list)
    counts = Counter()
    for i in range(n):
        plan = sample_plan(rng, planning_style='threaded')
        doc = realize(plan, seed=seed+i, lexical_exponent=1.05,
                      surface_version=surface_version)
        by_family = defaultdict(list)
        for site in doc.sites:
            assert answer_from_visible_prefix(
                doc.tokens[:site.index], lexical_enabled=True) == site.answer
            counts[f'all_sites/{site.family}'] += 1
            by_family[site.family].append(site)
        for family, candidates in by_family.items():
            site = site_rng.choice(candidates)
            crop = window_for_site(doc, site, width)
            if crop is None:
                continue
            windows[family].append(crop)
            question = is_question_site(doc, site)
            question_windows[(family, 'question' if question else 'prose')].append(crop)
            counts[f'window_form/{"question" if question else "prose"}'] += 1
    eligible = [first_words(row['preview'], words) for row in rows[n:]
                if len(WORD.findall(row['preview'])) >= width]
    result = {}
    for family in sorted(windows):
        if len(windows[family]) > len(eligible):
            raise ValueError('Too few natural windows')
        ref_rng = random.Random(seed ^ int.from_bytes(
            hashlib.sha256(family.encode()).digest()[:8], 'big'))
        natural = []
        for text in ref_rng.sample(eligible, len(windows[family])):
            count = len(WORD.findall(text))
            center = ref_rng.randrange(width//2, count-width//2)
            natural.append(word_window(text, center, width))
        result[family] = {
            'n': len(windows[family]),
            'synthetic': structure(windows[family]),
            'natural_random': structure(natural),
            'question_n': len(question_windows[(family, 'question')]),
            'prose_n': len(question_windows[(family, 'prose')]),
            'question_form': (structure(question_windows[(family, 'question')])
                              if len(question_windows[(family, 'question')]) >= 10
                              else None),
            'prose_form': (structure(question_windows[(family, 'prose')])
                           if len(question_windows[(family, 'prose')]) >= 10
                           else None),
        }
    return {'config': {'preview': str(preview), 'documents': n,
                       'source_first_index': rows[0]['index'],
                       'source_last_index': rows[n-1]['index'],
                       'preview_sha256': hashlib.sha256(
                           preview.read_bytes()).hexdigest(),
                       'words': words, 'window_width': width, 'seed': seed,
                       'surface_version': surface_version},
            'counts': dict(counts), 'families': result,
            'scope': 'One randomly selected site per family per generated '
                     'document. Natural windows use different source rows '
                     'from the same preview panel and random positions; '
                     'they are not task-matched. Question/prose splits are '
                     'descriptive, not a causal comparison. Every synthetic '
                     'site passed the visible-prefix answer checker. '
                     'No raw text is saved.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preview', type=Path, required=True)
    p.add_argument('--documents', type=int, default=512)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--width', type=int, default=41)
    p.add_argument('--seed', type=int, default=48901)
    p.add_argument('--surface-version', type=int, choices=(3, 4), default=3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.preview, a.documents, a.words, a.width, a.seed,
                   a.surface_version)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for family, row in result['families'].items():
        print(family, row['n'], row['question_n'],
              round(row['synthetic'][
                  'repeated_across_docs_trigram_occurrence_share'], 3),
              round(row['natural_random'][
                  'repeated_across_docs_trigram_occurrence_share'], 3),
              flush=True)


if __name__ == '__main__':
    main()

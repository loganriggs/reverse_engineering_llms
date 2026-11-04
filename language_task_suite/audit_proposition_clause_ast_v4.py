"""Paired V3/V4 clause-form and visible-semantics audit, aggregate only."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from audit_document_structure_vs_fineweb import structure
from audit_semantic_syntax_opportunities import first_words
from document_proposition_clause_ast import parse_prefix
from document_proposition_plan import edit_initial_fact, sample_plan
from document_proposition_surface import (
    answer_from_visible_prefix, parse_visible_document, realize,
)
from document_world import VALUES


def audit(preview: Path, n: int, words: int, seed: int):
    rows = json.loads(preview.read_text())
    if len(rows) < n:
        raise ValueError('Preview is too short')
    rng = random.Random(seed)
    counts = Counter()
    corpora = {'v3': [], 'v4': [], 'natural': []}
    for i in range(n):
        plan = sample_plan(rng, planning_style='threaded')
        v4_doc = None
        for version in (3, 4):
            doc = realize(plan, seed=seed+i, lexical_exponent=1.05,
                          surface_version=version)
            parse_visible_document(doc.tokens, lexical_enabled=True)
            corpora[f'v{version}'].append(first_words(doc.text(), words))
            counts[f'v{version}_sites'] += len(doc.sites)
            if version != 4:
                continue
            v4_doc = doc
            for site in doc.sites:
                assert doc.tokens[site.index] == site.answer
                assert answer_from_visible_prefix(
                    doc.tokens[:site.index], lexical_enabled=True
                ) == site.answer
                counts[f'site_{site.family}'] += 1
                if site.family not in ('comparison', 'fact', 'update'):
                    continue
                start = max((j+1 for j, token in enumerate(
                    doc.tokens[:site.index]) if token == '.'), default=0)
                lead = [token for token in doc.tokens[start:site.index]
                        if token != '\n']
                clause = parse_prefix(lead)
                if clause is None:
                    continue  # Legacy question-form fact/update sites.
                frame, _ = clause
                assert frame.kind == ('comparison' if site.family ==
                                      'comparison' else 'fact')
                counts[f'frame_{site.family}_{frame.prefix}'] += 1
                false_answer = next(candidate for candidate in site.candidates
                                    if candidate != site.answer)
                altered = list(doc.tokens)
                altered[site.index] = false_answer
                try:
                    parse_visible_document(altered, lexical_enabled=True)
                except (AssertionError, KeyError, ValueError, IndexError):
                    counts['false_claims_rejected'] += 1
                else:
                    raise AssertionError('False clause passed visible parser')
        assert v4_doc is not None
        # Intervene on a sampled opening value, repair dependent claims in the
        # plan, then verify that visible reconstruction tracks every changed
        # answer. Some plans have no supported change for this chosen value.
        obj = plan['moves'][0]['entity']
        attr = 'color'
        original = plan['initial'][obj][attr]
        for candidate in VALUES[attr]:
            if candidate == original:
                continue
            try:
                edited = edit_initial_fact(plan, obj, attr, candidate)
            except AssertionError:
                continue
            changed = realize(edited, seed=seed+i, lexical_exponent=1.05,
                              surface_version=4)
            parse_visible_document(changed.tokens, lexical_enabled=True)
            assert len(changed.sites) == len(v4_doc.sites)
            flips = 0
            for before, after in zip(v4_doc.sites, changed.sites):
                assert before.family == after.family
                assert answer_from_visible_prefix(
                    changed.tokens[:after.index], lexical_enabled=True
                ) == after.answer
                flips += before.answer != after.answer
            counts['supported_initial_color_edits'] += 1
            counts['changed_scored_answers_after_edit'] += flips
            break
        corpora['natural'].append(first_words(rows[i]['preview'], words))
    return {
        'config': {'source_first_index': rows[0]['index'],
                   'source_last_index': rows[n-1]['index'],
                   'preview_sha256': hashlib.sha256(
                       preview.read_bytes()).hexdigest(),
                   'documents': n, 'words': words, 'seed': seed},
        'counts': dict(counts),
        'form': {name: structure(texts) for name, texts in corpora.items()},
        'scope': 'Same sampled plans and lexicalized surfaces for V3/V4; '
                 'natural first-word windows from a previously inspected '
                 'FineWeb range. Every V4 scored site passed the visible '
                 'prefix checker. All new clause sites had their answer '
                 'replaced with a different candidate and rejected by the '
                 'visible-text parser. Supported opening-color edits had '
                 'their dependent claims repaired and visible answers '
                 'rechecked. Frame counts are by prefix tokens. '
                 'No raw source or generated text is saved.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preview', type=Path, required=True)
    p.add_argument('--documents', type=int, default=512)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--seed', type=int, default=178901)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.preview, a.documents, a.words, a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for name, row in result['form'].items():
        print(name,
              round(row['repeated_across_docs_trigram_occurrence_share'], 3),
              round(row['top10_word_share'], 3), flush=True)
    print('false claims rejected', result['counts']['false_claims_rejected'])


if __name__ == '__main__':
    main()

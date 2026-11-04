"""Audit source selection, joint gates, form, and bracket counterfactuals."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

from audit_document_structure_vs_fineweb import structure
from audit_rule_window_form import word_window
from audit_semantic_syntax_opportunities import WORD, first_words
from corpus_rule_mixture_dgp import Config, FAMILIES, draw, select_rows
from corpus_rule_reference_dgp import check_document


def _aggregate(texts):
    return {key: value for key, value in structure(texts).items()
            if key not in ('top10_words', 'top_sentence_starts')}


def _site_window(text, site, width=41):
    position = len(WORD.findall(text[:site.target[0]]))
    return word_window(text, position, width)


def _bracket_counterfactual(doc):
    bracket = next((s for s in doc.sites if s.kind == 'bracket'), None)
    if bracket is None:
        return None
    a, b = bracket.source[0], bracket.target[0]
    original = doc.text[a], doc.text[b]
    replacement = {('(', ')'): ('[', ']'), ('[', ']'): ('(', ')')}[original]
    chars = list(doc.text)
    chars[a], chars[b] = replacement
    new_site = replace(bracket, answer=replacement[1])
    sites = tuple(new_site if s is bracket else s for s in doc.sites)
    changed = replace(doc, text=''.join(chars), sites=sites)
    check_document(changed)
    assert changed.text[bracket.target[0]] != doc.text[bracket.target[0]]
    return changed


def audit(preview: Path, n: int, policy: str, gate_rate: float,
          coupling_strength: float, seed: int,
          excluded_pair: tuple[str, str] | None = None):
    rows = json.loads(preview.read_text())
    source_rng = random.Random(seed)
    selected, available = select_rows(rows, n, source_rng, policy)
    config = Config(gate_rate, coupling_strength, policy, excluded_pair)
    counts = Counter()
    eligible_pair = Counter()
    applied_pair = Counter()
    source_texts, transformed_texts = [], []
    windows = defaultdict(list)
    source_windows = defaultdict(list)
    for row in selected:
        item = draw(row, random.Random(seed*1000003 + row['index']), config)
        doc = item.document
        check_document(doc)
        counts['gate_rejections'] += item.gate_rejections
        if excluded_pair is not None:
            assert not set(excluded_pair) <= set(doc.applied)
        changed_chars = sum(a != b for a, b in zip(
            doc.source_text, doc.text))
        counts['source_characters'] += len(doc.source_text)
        counts['changed_characters'] += changed_chars
        counts[f'changed_characters_per_document/{changed_chars}'] += 1
        counts[f'eligible_count/{len(set(doc.eligible)&set(FAMILIES))}'] += 1
        counts[f'applied_count/{len(doc.applied)}'] += 1
        counts[f'latent_state/{item.rich_state}'] += 1
        for family in doc.eligible:
            if family in FAMILIES:
                counts[f'eligible/{family}'] += 1
        for family in doc.applied:
            counts[f'applied/{family}'] += 1
        for i, left in enumerate(FAMILIES):
            for right in FAMILIES[i+1:]:
                key = f'{left}+{right}'
                if left in doc.eligible and right in doc.eligible:
                    eligible_pair[key] += 1
                    if left in doc.applied and right in doc.applied:
                        applied_pair[key] += 1
        for site in doc.sites:
            counts[f'sites/{site.kind}'] += 1
            crop = _site_window(doc.text, site)
            if crop is not None:
                windows[site.kind].append(crop)
                source_crop = _site_window(doc.source_text, site)
                assert source_crop is not None
                source_windows[site.kind].append(source_crop)
        edited = _bracket_counterfactual(doc)
        if edited:
            counts['valid_bracket_counterfactuals'] += 1
        source_texts.append(first_words(doc.source_text, 100))
        transformed_texts.append(first_words(doc.text, 100))
    assert len({row['sha256'] for row in selected}) == n
    return {
        'config': {'source_first_index': rows[0]['index'],
                   'source_last_index': rows[-1]['index'],
                   'preview_sha256': hashlib.sha256(
                       preview.read_bytes()).hexdigest(),
                   'selected_unique_sources': n,
                   'available_unique_sources_under_policy': available,
                   'source_policy': policy, 'gate_rate': gate_rate,
                   'coupling_strength': coupling_strength, 'seed': seed,
                   'excluded_pair': excluded_pair},
        'counts': dict(counts),
        'pairwise_joint_eligibility': dict(eligible_pair),
        'pairwise_joint_application': dict(applied_pair),
        'whole_document_form': {'source': _aggregate(source_texts),
                                'transformed': _aggregate(transformed_texts)},
        'site_window_form': {kind: _aggregate(rows)
                             for kind, rows in windows.items()
                             if len(rows) >= 10},
        'paired_source_site_window_form': {
            kind: _aggregate(rows) for kind, rows in source_windows.items()
            if len(rows) >= 10},
        'scope': 'Empirical source text is sampled without replacement. '
                 'A binary document latent state jointly changes gate '
                 'probabilities while keeping their marginal rates fixed. '
                 'The source corpus is an empirical component of this DGP; '
                 'background tokens have no symbolic next-token oracle. '
                 'All sampled documents and bracket opener/closer '
                 'counterfactuals pass the local rule checker. Site-window '
                 'form is paired with the same unedited source spans; this '
                 'is not a population-matched natural task baseline. '
                 'No source or generated prose is saved.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preview', type=Path, required=True)
    p.add_argument('--documents', type=int, required=True)
    p.add_argument('--source-policy', choices=('uniform','multitask'),
                   required=True)
    p.add_argument('--gate-rate', type=float, default=.5)
    p.add_argument('--coupling-strength', type=float, default=0.)
    p.add_argument('--exclude-pair', nargs=2, choices=FAMILIES)
    p.add_argument('--seed', type=int, default=91337)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.preview, a.documents, a.source_policy,
                   a.gate_rate, a.coupling_strength, a.seed,
                   tuple(a.exclude_pair) if a.exclude_pair else None)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['config']['available_unique_sources_under_policy'],
          {k:v for k,v in result['counts'].items()
           if k.startswith('sites/')},
          result['counts'].get('valid_bracket_counterfactuals',0),
          flush=True)


if __name__ == '__main__':
    main()

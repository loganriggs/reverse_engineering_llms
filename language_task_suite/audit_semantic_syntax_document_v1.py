"""Semantic interventions and independent form audit for the tree DGP."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import spacy
from tokenizers import Tokenizer

from audit_document_structure_vs_fineweb import structure
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_rule_window_form import word_window
from audit_semantic_syntax_opportunities import summarize
from document_world_v9_typed_lexicon import COLOR_CONSONANT, COLOR_VOWEL, PEOPLE
from semantic_syntax_document_v1 import (
    Document, check_document, edit_last_event, realize, replay_visible, sample,
)


CONFIGS = {
    'simple': dict(lexical_exponent=0., source_coupling=0.,
                   coordination_rate=0., adjunct_rate=0.,
                   relative_rate=0., intro_interleave_rate=0.,
                   recap_interleave_rate=0.),
    'rich_independent': dict(lexical_exponent=1., source_coupling=0.,
                             coordination_rate=.4, adjunct_rate=.25,
                             relative_rate=.18, intro_interleave_rate=.6,
                             recap_interleave_rate=.6),
    'rich_coupled': dict(lexical_exponent=1., source_coupling=.9,
                         coordination_rate=.4, adjunct_rate=.25,
                         relative_rate=.18, intro_interleave_rate=.6,
                         recap_interleave_rate=.6),
}
RENDER_KEYS = ('coordination_rate', 'adjunct_rate', 'relative_rate',
               'intro_interleave_rate', 'recap_interleave_rate')
WORDS = __import__('re').compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def first_words(text, count):
    matches = list(WORDS.finditer(text))
    if len(matches) < count:
        return None
    return text[:matches[count-1].end()]


def intervention_checks(doc: Document, cfg):
    color_site = next(site for site in doc.sites
                      if site.family == 'current_color')
    source_site = next(site for site in doc.sites
                       if site.family == 'latest_source')
    color_event = next(event for event in reversed(doc.world['events'])
                       if event.entity == color_site.entity)
    alt_color = next(color for color in COLOR_CONSONANT + COLOR_VOWEL
                     if color not in (color_event.old, color_event.new))
    edited = edit_last_event(doc.world, color_site.entity,
                             new_color=alt_color)
    rerender = realize(edited, seed=doc.world['render_seed'],
                       **{k: cfg[k] for k in RENDER_KEYS})
    by_family = {site.family: site.answer for site in rerender.sites}
    assert by_family['current_color'] == alt_color
    assert by_family['latest_source'] == source_site.answer

    source_event = next(event for event in reversed(doc.world['events'])
                        if event.entity == source_site.entity)
    alt_source = next(name for name in PEOPLE
                      if name != source_event.source)
    edited = edit_last_event(doc.world, source_site.entity,
                             new_source=alt_source)
    rerender = realize(edited, seed=doc.world['render_seed'],
                       **{k: cfg[k] for k in RENDER_KEYS})
    by_family = {site.family: site.answer for site in rerender.sites}
    assert by_family['latest_source'] == alt_source
    assert by_family['current_color'] == color_site.answer

    for site, replacement in ((color_site, alt_color),
                              (source_site, alt_source)):
        tokens = list(doc.tokens)
        tokens[site.index] = replacement
        corrupted = Document(tuple(tokens), doc.sites,
                             doc.world, doc.trees)
        try:
            replay_visible(corrupted.tokens)
        except ValueError:
            continue
        raise AssertionError('False visible recap was accepted')
    return True


def audit(preview: Path, documents=512, seed=74211, target_words=60):
    rows = json.loads(preview.read_text())
    if len(rows) < documents:
        raise ValueError('Too few natural previews')
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    nlp = spacy.load('en_core_web_sm')
    eligible = [row['preview'] for row in rows
                if len(WORDS.findall(row['preview'])) >= target_words]
    if len(eligible) < documents:
        raise ValueError('Too few long natural previews')
    rng = random.Random(seed)
    natural = [first_words(text, target_words)
               for text in rng.sample(eligible, documents)]
    natural_windows = []
    for text in rng.sample(eligible, documents):
        count = len(WORDS.findall(text))
        center = rng.randrange(20, count-20)
        natural_windows.append(word_window(text, center, 41))
    result = {'config': {'seed': seed, 'documents': documents,
                         'target_words': target_words,
                         'natural_first_index': rows[0]['index'],
                         'natural_last_index': rows[-1]['index'],
                         'arms': CONFIGS},
              'natural': {'form': structure(natural),
                          'syntax': summarize(nlp, natural, target_words),
                          'random_41_word_window':
                              structure(natural_windows)},
              'arms': {},
              'caveat': 'Synthetic world is truth-checked and syntax is '
                        'compositional, but natural-form measures may still '
                        'reject this vertical slice. Natural windows are '
                        'random rather than rule-matched. No source prose '
                        'is exported.'}
    for arm, cfg in CONFIGS.items():
        texts, crops, examples = [], [], []
        windows = {'current_color': [], 'latest_source': []}
        counts, tree_ops = Counter(), Counter()
        for i in range(documents):
            doc = sample(seed+100000+i, **cfg)
            assert check_document(doc)
            intervention_checks(doc, cfg)
            counts['documents'] += 1
            counts['sites'] += len(doc.sites)
            counts['fits_256_bpe'] += len(tokenizer.encode(doc.text()).ids) <= 256
            for tree in doc.trees:
                stack = [tree]
                while stack:
                    node = stack.pop()
                    tree_ops[node.op] += 1
                    stack.extend(node.children)
            text = doc.text()
            texts.append(text)
            for site in doc.sites:
                target_word = len(WORDS.findall(' '.join(
                    doc.tokens[:site.index])))
                window = word_window(text, target_word, 41)
                if window is not None:
                    windows[site.family].append(window)
            crop = first_words(text, target_words)
            if crop is not None:
                crops.append(crop)
            if len(examples) < 4:
                examples.append(text)
        result['arms'][arm] = {
            'counts': dict(counts), 'tree_ops': dict(tree_ops),
            'first_words_eligible': len(crops),
            'form': structure(crops),
            'syntax': summarize(nlp, crops, target_words),
            'site_windows': {family: {'n': len(rows),
                                      'form': structure(rows)}
                             for family, rows in windows.items() if rows},
            'examples': examples}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', type=Path, required=True)
    parser.add_argument('--documents', type=int, default=512)
    parser.add_argument('--seed', type=int, default=74211)
    parser.add_argument('--words', type=int, default=60)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = audit(args.preview, args.documents, args.seed, args.words)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    for name, arm in out['arms'].items():
        form = arm['form']
        print(name, arm['counts'], 'repeat3',
              round(form['repeated_across_docs_trigram_occurrence_share'], 3),
              'syntax types', arm['syntax']['distinct_dependency_signatures'],
              flush=True)


if __name__ == '__main__':
    main()

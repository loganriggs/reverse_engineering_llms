"""Dependency-level form and semantic-frame opportunity screen.

This is an aggregate-only development diagnostic. A dependency match is an
opportunity for a typed renderer, not a verified truth-conditional template.
Run separately on source-disjoint local FineWeb preview panels.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

import spacy

from document_proposition_plan import sample_plan
from document_proposition_surface import realize
from audit_document_structure_vs_fineweb import structure
from audit_rule_window_form import word_window
from document_proposition_coordination import (
    answer_from_composed_prefix, compose, expand_visible, parse_composed,
)
from document_proposition_relative import (
    add_relative_clauses, answer_from_relative_prefix,
    expand_checked, parse_relative,
)


WORD = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")
REPORTING = frozenset(('say', 'report', 'write', 'announce', 'confirm',
                       'claim', 'tell', 'note', 'state', 'explain'))
SUBJECT_DEPS = frozenset(('nsubj', 'nsubjpass'))
OBJECT_DEPS = frozenset(('dobj', 'obj'))
FAMILIES = ('adjective_predication', 'simple_adjective_predication',
            'transitive_event', 'simple_transitive_event', 'reporting_clause',
            'relative_clause', 'adverbial_clause', 'complement_clause',
            'coordination', 'negation', 'question')


def first_words(text, count):
    words = list(WORD.finditer(text))
    return text[:words[count - 1].end()] if len(words) >= count else text


def clause_signature(sent):
    root = sent.root
    children = Counter(child.dep_ for child in root.children
                       if child.i >= sent.start and child.i < sent.end)
    # Structure only: no source lexemes, which would make every signature
    # trivially distinct and would risk storing corpus snippets.
    deps = tuple(f'{dep}:{min(count, 2)}' for dep, count in sorted(children.items())
                 if dep not in ('punct', 'det'))
    return '|'.join((root.pos_, *deps))


def classify(sent):
    root = sent.root
    children = list(root.children)
    labels = {child.dep_ for child in children}
    negated = any(token.dep_ == 'neg' for token in sent)
    subjects = [token for token in children if token.dep_ in SUBJECT_DEPS]
    objects = [token for token in children if token.dep_ in OBJECT_DEPS]
    copula = [child for child in children if child.dep_ == 'cop'
              and child.lemma_ == 'be']
    complements = [child for child in children if child.dep_ == 'acomp'
                   and child.pos_ == 'ADJ' and child.tag_ not in ('JJR', 'JJS')]
    # Conservatively require one overt subject and a simple adjectival
    # predicate. Still only a parsing opportunity, not a safe substitution.
    predication = (len(subjects) == 1 and not negated and
                   ((root.lemma_ == 'be' and root.pos_ in ('AUX', 'VERB')
                     and len(complements) == 1) or
                    (root.pos_ == 'ADJ' and bool(copula) and
                     root.tag_ not in ('JJR', 'JJS'))))
    transitive = (root.pos_ == 'VERB' and len(subjects) == 1 and
                  bool(objects) and not negated)
    reporting = (root.lemma_.lower() in REPORTING and bool(subjects) and
                 bool(labels & {'ccomp', 'dobj', 'obj', 'parataxis'}))
    embedded = any(t.dep_ in ('relcl', 'acl:relcl', 'advcl', 'ccomp',
                              'xcomp', 'conj') for t in sent)
    short = len(sent) <= 18
    complete_statement = sent.text.strip().endswith(('.', '!'))
    nominal_subject = (len(subjects) == 1 and
                       subjects[0].pos_ in ('NOUN', 'PROPN'))
    return {
        'adjective_predication': predication,
        'simple_adjective_predication': (
            predication and short and complete_statement and
            nominal_subject and not embedded and
            not any(t.pos_ == 'VERB' for t in sent)),
        'transitive_event': transitive,
        'simple_transitive_event': (
            transitive and len(objects) == 1 and short and
            complete_statement and nominal_subject and not embedded and
            sum(t.pos_ == 'VERB' for t in sent) == 1),
        'reporting_clause': reporting,
        'relative_clause': any(t.dep_ in ('relcl', 'acl:relcl') for t in sent),
        'adverbial_clause': any(t.dep_ == 'advcl' for t in sent),
        'complement_clause': any(t.dep_ in ('ccomp', 'xcomp') for t in sent),
        'coordination': any(t.dep_ == 'conj' for t in sent),
        'negation': negated,
        'question': any(t.text == '?' for t in sent),
    }


def summarize(nlp, texts, target_words):
    counts, signatures = Counter(), Counter()
    lengths = []
    input_lengths = [len(WORD.findall(text)) for text in texts]
    for doc in nlp.pipe(texts, batch_size=32):
        counts['documents'] += 1
        doc_families = set()
        for sent in doc.sents:
            nwords = len(WORD.findall(sent.text))
            if not 5 <= nwords <= 60:
                counts['discarded_segments'] += 1
                continue
            counts['sentences'] += 1
            lengths.append(nwords)
            tags = classify(sent)
            for name, matched in tags.items():
                if matched:
                    counts[f'sentences/{name}'] += 1
                    doc_families.add(name)
            signatures[clause_signature(sent)] += 1
        for name in doc_families:
            counts[f'documents/{name}'] += 1
        counts[f'doc_families/{len(doc_families)}'] += 1
    n = counts['sentences']
    return {
        'documents': counts['documents'], 'sentences': n,
        'mean_input_words': sum(input_lengths) / len(input_lengths),
        'documents_below_target_words': sum(
            length < target_words for length in input_lengths),
        'discarded_segments': counts['discarded_segments'],
        'mean_sentence_words': sum(lengths) / n,
        'sentence_words_p50': sorted(lengths)[n // 2],
        'distinct_dependency_signatures': len(signatures),
        'top10_signature_share': sum(v for _, v in signatures.most_common(10)) / n,
        'top10_signature_counts': [v for _, v in signatures.most_common(10)],
        'families': {name: {
            'sentences': counts[f'sentences/{name}'],
            'documents': counts[f'documents/{name}'],
        } for name in FAMILIES},
        'family_count_per_document': {
            key.removeprefix('doc_families/'): value
            for key, value in counts.items()
            if key.startswith('doc_families/')},
    }


def parser_self_check(nlp):
    cases = (
        ('The report is quiet.', 'adjective_predication', True),
        ('The report is not quiet.', 'adjective_predication', False),
        ('Alice moved the crate.', 'transitive_event', True),
        ('The crate that Alice moved was heavy.', 'relative_clause', True),
    )
    for sentence, family, expected in cases:
        got = classify(next(nlp(sentence).sents))[family]
        if got != expected:
            raise AssertionError((sentence, family, expected, got))


def site_in_coordination(tokens, position):
    start = position - 1
    while start >= 0 and tokens[start] != '.':
        start -= 1
    end = position
    while end < len(tokens) and tokens[end] != '.':
        end += 1
    return any(tokens[i:i+2] == (',', 'and')
               for i in range(start+1, end-1))


def site_in_relative(tokens, position):
    start = position - 1
    while start >= 0 and tokens[start] != '.':
        start -= 1
    return any(tokens[i:i+2] == (',', 'which')
               for i in range(start+1, position-1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preview', type=Path, required=True)
    p.add_argument('--documents', type=int, default=512)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--seed', type=int, default=46601)
    p.add_argument('--coordination-rate', type=float, default=.75)
    p.add_argument('--allow-scored-joins', action='store_true')
    p.add_argument('--surface-operator', choices=('coordination', 'relative'),
                   default='coordination')
    p.add_argument('--relative-rate', type=float, default=.75)
    p.add_argument('--window-width', type=int, default=41)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.documents <= 0 or a.words <= 0:
        raise ValueError('Positive document and word counts required')
    rows = json.loads(a.preview.read_text())
    if len(rows) < a.documents:
        raise ValueError('Too few preview documents')
    natural_texts = [first_words(row['preview'], a.words)
                     for row in rows[:a.documents]]
    rng = random.Random(a.seed)
    site_rng = random.Random(a.seed ^ 0x51E5)
    generated_texts, composed_texts = [], []
    task_windows = {'synthetic': [], 'composed': []}
    task_families = Counter()
    joined = validated_sites = coordinated_sites = 0
    for index in range(a.documents):
        plan = sample_plan(rng, planning_style='threaded')
        generated = realize(plan, seed=a.seed + index,
                            lexical_exponent=1.05, surface_version=3)
        generated_texts.append(first_words(generated.text(), a.words))
        if a.surface_operator == 'coordination':
            composed, count = compose(
                generated, seed=a.seed + index + 800000,
                rate=a.coordination_rate, allow_scored=a.allow_scored_joins)
            assert expand_visible(composed.tokens) == generated.tokens
            parse_composed(composed, lexical_enabled=True)
            answer = answer_from_composed_prefix
            modified_site = site_in_coordination
        else:
            composed, count = add_relative_clauses(
                generated, seed=a.seed + index + 800000,
                rate=a.relative_rate, lexical_enabled=True)
            assert expand_checked(composed.tokens, True) == generated.tokens
            parse_relative(composed, lexical_enabled=True)
            answer = answer_from_relative_prefix
            modified_site = site_in_relative
        for site in composed.sites:
            assert answer(
                composed.tokens[:site.index], lexical_enabled=True) == site.answer
            validated_sites += 1
            coordinated_sites += modified_site(composed.tokens, site.index)
        joined += count
        composed_texts.append(first_words(composed.text(), a.words))
        if generated.sites:
            index = site_rng.randrange(len(generated.sites))
            site, joined_site = generated.sites[index], composed.sites[index]
            original_word = len(WORD.findall(' '.join(
                generated.tokens[:site.index])))
            joined_word = len(WORD.findall(' '.join(
                composed.tokens[:joined_site.index])))
            original_window = word_window(generated.text(), original_word,
                                          a.window_width)
            joined_window = word_window(composed.text(), joined_word,
                                        a.window_width)
            if original_window and joined_window:
                task_windows['synthetic'].append(original_window)
                task_windows['composed'].append(joined_window)
                task_families[site.family] += 1
    eligible_natural = [first_words(row['preview'], a.words)
                        for row in rows if len(WORD.findall(
                            row['preview'])) >= a.window_width]
    if len(eligible_natural) < len(task_windows['synthetic']):
        raise ValueError('Too few natural rule-window references')
    natural_rng = random.Random(a.seed ^ 0x572A)
    natural_windows = []
    for text in natural_rng.sample(eligible_natural,
                                  len(task_windows['synthetic'])):
        total = len(WORD.findall(text))
        center = natural_rng.randrange(a.window_width//2,
                                       total-a.window_width//2)
        natural_windows.append(word_window(text, center, a.window_width))
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    parser_self_check(nlp)
    natural = summarize(nlp, natural_texts, a.words)
    generated = summarize(nlp, generated_texts, a.words)
    composed = summarize(nlp, composed_texts, a.words)
    report = {
        'config': {'preview': str(a.preview), 'documents': a.documents,
                   'words': a.words, 'seed': a.seed,
                   'source_first_index': rows[0]['index'],
                   'source_last_index': rows[a.documents - 1]['index'],
                   'preview_sha256': hashlib.sha256(a.preview.read_bytes()).hexdigest(),
                   'parser': 'spaCy en_core_web_sm',
                   'spacy_version': spacy.__version__,
                   'model_version': nlp.meta.get('version'),
                   'synthetic': 'threaded proposition surface v3',
                   'coordination_rate': a.coordination_rate,
                   'allow_scored_joins': a.allow_scored_joins,
                   'surface_operator': a.surface_operator,
                   'relative_rate': a.relative_rate},
        'natural': natural, 'synthetic': generated,
        'composed': composed,
        'composition_checks': {'full_document_reversible': a.documents,
                               'visible_parser_passed': a.documents,
                               'visible_answer_sites_passed': validated_sites,
                               'sites_inside_modified_sentences': coordinated_sites,
                               'operator_applications': joined,
                               'joined_sentence_pairs': (
                                   joined if a.surface_operator == 'coordination'
                                   else 0),
                               'relative_clauses': (
                                   joined if a.surface_operator == 'relative'
                                   else 0)},
        'task_window': {
            'width_words': a.window_width,
            'n': len(task_windows['synthetic']),
            'sampled_family_counts': dict(task_families),
            'natural': structure(natural_windows),
            'synthetic': structure(task_windows['synthetic']),
            'composed': structure(task_windows['composed']),
        },
        'caveat': 'Dependency labels and simple pattern matches are imperfect. '
                  'Adjective/transitive/reporting counts are opportunities, '
                  'not certified semantics or ready-to-sample templates. '
                  'Both panels use first-word crops; web markup and parser '
                  'errors may distort sentence counts. The natural panel '
                  'is a development reference, not a population estimate. '
                  'No source text is saved.',
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    for name in ('natural', 'synthetic', 'composed'):
        row = report[name]
        print(name, row['sentences'], row['distinct_dependency_signatures'],
              round(row['top10_signature_share'], 3),
              {key: value['sentences'] for key, value in row['families'].items()},
              flush=True)


if __name__ == '__main__':
    main()

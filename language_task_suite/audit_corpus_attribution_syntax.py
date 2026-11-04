"""Screen natural reporting clauses for reusable *syntax*, not claims.

This does not produce a training DGP. It asks whether a corpus supplies enough
structurally distinct, locally substitutable attribution frames to justify a
corpus-calibrated semantic renderer. No source sentences are written to disk.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import spacy


REPORTING = frozenset({
    'say', 'report', 'claim', 'state', 'explain', 'confirm', 'write',
    'announce', 'argue', 'suggest', 'note', 'tell', 'add',
})
BAD_OUTSIDE = frozenset({'he', 'she', 'they', 'his', 'her', 'their', 'this',
                         'these', 'those', 'it', 'its'})
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _span_indices(token):
    indices = sorted(child.i for child in token.subtree)
    if indices != list(range(indices[0], indices[-1] + 1)):
        return None
    return indices[0], indices[-1] + 1


def extract(sent):
    """Return a conservative abstract frame or a rejection reason."""
    root = sent.root
    if root.lemma_.lower() not in REPORTING or root.pos_ != 'VERB':
        return None, 'not_reporting_root'
    subjects = [t for t in root.children if t.dep_ in ('nsubj', 'nsubjpass')]
    complements = [t for t in root.children if t.dep_ == 'ccomp']
    if len(subjects) != 1 or len(complements) != 1:
        return None, 'not_single_subject_and_clause'
    subject, complement = subjects[0], complements[0]
    if subject.pos_ not in ('PROPN', 'NOUN'):
        return None, 'subject_not_nominal'
    subject_span = _span_indices(subject)
    clause_span = _span_indices(complement)
    if subject_span is None or clause_span is None:
        return None, 'noncontiguous_subtree'
    ss, se = subject_span
    cs, ce = clause_span
    if not (sent.start <= ss < se <= root.i < cs < ce <= sent.end):
        return None, 'unsupported_order'
    if ce - cs < 3 or ce - cs > 24 or se - ss > 6:
        return None, 'slot_length'
    if any(t.ent_type_ in ('PERSON', 'ORG', 'GPE', 'DATE', 'TIME', 'MONEY')
           or t.like_num or t.lower_ in BAD_OUTSIDE
           for t in sent if not (ss <= t.i < se or cs <= t.i < ce)):
        return None, 'context_dependent_shell'
    # A complement headed by "that" can remain in the slot; the new clause
    # should be substituted as an entire proposition with the same marker.
    shell = tuple('{speaker}' if i == ss else
                  '{claim}' if i == cs else sent.doc[i].text
                  for i in range(sent.start, sent.end)
                  if not (ss < i < se or cs < i < ce))
    if shell.count('{speaker}') != 1 or shell.count('{claim}') != 1:
        return None, 'slot_error'
    # Syntax-only signature prevents a unique lexical shell from looking like
    # evidence of a diverse grammar.
    signature = (root.lemma_.lower(),
                 tuple((t.dep_, t.pos_) for t in sent
                       if t.i not in range(ss, se)
                       and t.i not in range(cs, ce)))
    return (shell, signature, root.lemma_.lower()), None


def audit(preview: Path, limit: int):
    if limit <= 0:
        raise ValueError('limit must be positive')
    rows = json.loads(preview.read_text())[:limit]
    if not rows:
        raise ValueError('preview has no rows')
    nlp = spacy.load('en_core_web_sm')
    accepted, reason = extract(next(nlp('Alice said that the map was blue.').sents))
    assert reason is None and accepted[0] == (
        '{speaker}', 'said', '{claim}', '.')
    rejected, reason = extract(next(nlp(
        'Alice reported on Tuesday that the map was blue.').sents))
    assert rejected is None and reason == 'context_dependent_shell'
    counts = Counter()
    signature_counts = Counter()
    shell_counts = Counter()
    verb_counts = Counter()
    by_document = Counter()
    for index, doc in enumerate(nlp.pipe((row['preview'] for row in rows),
                                         batch_size=32)):
        accepted = 0
        for sent in doc.sents:
            if not (5 <= len(WORD.findall(sent.text)) <= 50):
                continue
            counts['sentences_screened'] += 1
            result, reason = extract(sent)
            if result is None:
                counts[f'reject/{reason}'] += 1
                continue
            shell, signature, verb = result
            accepted += 1
            counts['accepted_sentences'] += 1
            if (len(shell) in (4, 5) and shell[0] == '{speaker}'
                    and shell[2] == '{claim}' and shell[3] in ('.', '!')
                    and (len(shell) == 4 or shell[4].isspace())):
                counts['bare_reporting_shell'] += 1
            if any(token in ('"', '“', '”', '‘', '’') for token in shell):
                counts['shell_with_quote_mark'] += 1
            if any('\n' in token for token in shell):
                counts['shell_with_newline'] += 1
            signature_counts[str(signature)] += 1
            shell_counts[' '.join(shell)] += 1
            verb_counts[verb] += 1
        by_document[accepted] += 1
    return {
        'scope': {'preview_sha256': hashlib.sha256(preview.read_bytes()).hexdigest(),
                  'first_index': rows[0]['index'],
                  'last_index': rows[-1]['index'], 'documents': len(rows),
                  'screen': 'root reporting verb + overt nominal subject + one ccomp; '
                            'no external named entities, numbers, or demonstratives'},
        'counts': dict(counts),
        'accepted_by_document': dict(by_document),
        'distinct_syntax_signatures': len(signature_counts),
        'top10_signature_share': (sum(v for _, v in signature_counts.most_common(10)) /
                                  max(1, counts['accepted_sentences'])),
        'distinct_surface_shells': len(shell_counts),
        'top10_shell_share': (sum(v for _, v in shell_counts.most_common(10)) /
                              max(1, counts['accepted_sentences'])),
        'verb_counts': dict(verb_counts),
        'caveat': 'Parser-screened syntactic opportunities only; semantic '
                  'substitution and naturalness are not established. No '
                  'source text or unique frames are saved.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=2000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.preview, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['scope'], result['counts']['accepted_sentences'],
          result['distinct_syntax_signatures'],
          result['distinct_surface_shells'])


if __name__ == '__main__':
    main()

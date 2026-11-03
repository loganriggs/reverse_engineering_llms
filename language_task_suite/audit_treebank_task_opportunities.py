"""Count natural sentence-level task opportunities in a parsed treebank.

This is an opportunity screen, not a claim that any edit yields a valid
next-token oracle or that separately parsed sentences form documents.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_raw_pcfg_treebank_pilot import leaves, parse_tree
from audit_tree_substitution_pilot import walk
from treebank_agreement_dgp import subject_candidate, vp_candidate


def inspect(tree):
    words = tuple(leaves(tree))
    quotes = []
    pending = []
    for i, token in enumerate(words):
        if token == '``':
            pending.append(i)
        elif token == "''" and pending:
            quotes.append((pending.pop(), i))
    agreements = []
    for _, node in walk(tree):
        if node[0] != 'S':
            continue
        children = [child for child in node[1]
                    if not isinstance(child, str)]
        for left, right in zip(children, children[1:]):
            if left[0] != 'NP' or right[0] != 'VP':
                continue
            subject = subject_candidate(left)
            vp = vp_candidate(right)
            if subject and vp:
                agreements.append((subject, vp))
    content = [token.lower() for token in words if token.isalpha()
               and len(token) >= 4]
    repeated_content = len(set(content)) < len(content)
    return words, quotes, agreements, repeated_content


def audit(path, n):
    counts = Counter()
    quote_gaps = Counter()
    for line in path.open():
        if counts['sentences'] >= n:
            break
        words, quotes, agreements, repeated = inspect(parse_tree(line))
        counts['sentences'] += 1
        counts['tokens'] += len(words)
        counts['paired_quotes'] += len(quotes)
        counts['quote_sentences'] += bool(quotes)
        counts['agreement_sites'] += len(agreements)
        counts['agreement_sentences'] += bool(agreements)
        counts['agreement_attractor_sites'] += sum(
            subject[2] for subject, _ in agreements)
        counts['agreement_and_quote_sentences'] += bool(quotes and
                                                         agreements)
        counts['repeated_content_sentences'] += repeated
        for start, end in quotes:
            quote_gaps[min((end-start)//5*5, 100)] += 1
    return {'source': path.name, 'sentences_requested': n,
            'counts': dict(counts),
            'quote_gap_tokens_rounded_down_to_5': dict(quote_gaps),
            'definitions': {
                'agreement_site': 'An S node with adjacent NP subject and VP '
                                  'whose first word is tagged VBZ/VBP and '
                                  'belongs to an is/are, has/have, or '
                                  'does/do pair. Subject passes the same '
                                  'conservative filter as the DGP.',
                'attractor': 'The eligible subject NP contains a direct PP '
                             'with an opposite-number NP under the current '
                             'conservative head rule.',
                'paired_quote': 'A treebank opening `` and closing quote '
                                "'' in one parsed sentence.",
                'repeated_content': 'A case-insensitive alphabetic token of '
                                    'length at least four repeats within '
                                    'the sentence; this is a weak lexical '
                                    'proxy, not resolved coreference.'},
            'caveat': 'Counts are lexical/parse opportunities, not verified '
                      'semantics or frequencies for FineWeb. The treebank '
                      'does not provide a sampled document process.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--treebank', type=Path, required=True)
    p.add_argument('--sentences', type=int, default=100000)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.treebank, args.sentences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'])


if __name__ == '__main__':
    main()

"""Document-level FineWeb preview statistics; store hashes, never source prose.

Pass --preview-json for an already fetched temporary preview. Otherwise
stream the public sample-10BT corpus at --skip. Counts are surface cues,
not inferred semantic tasks.
"""

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from itertools import islice
from pathlib import Path
from urllib.parse import urlparse


WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
ILLUSTRATIVE = (23006, 23010, 23014, 23019, 23022, 23033,
                23038, 23043, 23045, 23059, 23062, 23063)


def fetch(skip, n):
    from datasets import load_dataset
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    rows = []
    for index, row in enumerate(islice(source, skip, skip+n), skip):
        text = row['text']
        rows.append(dict(index=index, url=row['url'], length=len(text),
                         sha256=hashlib.sha256(text.encode()).hexdigest(),
                         preview=text[:1200]))
    assert len(rows) == n
    return rows


def summarize(rows, skip, n):
    assert len(rows) == n
    assert [row['index'] for row in rows] == list(range(skip, skip+n))
    urls = [row['url'] for row in rows]
    assert len(set(urls)) == n
    previews = [row['preview'] for row in rows]
    eligible = [WORDS.findall(text)[:100] for text in previews
                if len(WORDS.findall(text)) >= 100]
    counts = Counter(word.lower() for words in eligible for word in words)
    total = sum(counts.values())
    assert total == 100*len(eligible)
    docs = lambda predicate: sum(bool(predicate(text)) for text in previews)
    return dict(
        source='HuggingFaceFW/fineweb sample-10BT train',
        selection=(f'{n} contiguous streaming documents at offsets '
                   f'{skip}..{skip+n-1}; first 1200 characters inspected '
                   'per document'),
        n_documents=n, unique_urls=len(set(urls)),
        unique_hosts=len({urlparse(url).hostname for url in urls}),
        median_full_document_characters=statistics.median(
            row['length'] for row in rows),
        first1200_diagnostics=dict(
            at_least_100_alphabetic_words=len(eligible),
            types_in_first100_words_of_eligible=len(counts),
            singletons_in_first100_words_of_eligible=sum(
                value == 1 for value in counts.values()),
            top10_share_in_first100_words_of_eligible=(
                sum(value for _, value in counts.most_common(10))/total),
            documents_with_newline=docs(lambda s: '\n' in s),
            documents_with_digit=docs(lambda s: bool(re.search(r'\d', s))),
            documents_with_pipe=docs(lambda s: '|' in s),
            documents_with_quote_mark=docs(lambda s: any(
                mark in s for mark in ('"', '“', '”'))),
            documents_with_question_mark=docs(lambda s: '?' in s),
            the_share_first100_eligible=counts['the']/total,
            a_share_first100_eligible=counts['a']/total,
            an_share_first100_eligible=counts['an']/total),
        document_sha256=[dict(offset=row['index'], sha256=row['sha256'])
                         for row in rows],
        illustrative_offsets=[i for i in ILLUSTRATIVE
                              if skip <= i < skip+n],
        caveat='Heuristic surface counts from only the first 1200 '
               'characters, not semantic-dependency prevalence. Text and '
               'URLs are omitted. Selected examples are illustrative and '
               'not a random sample of mechanism types.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=23000)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--preview-json', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = (json.loads(args.preview_json.read_text())
            if args.preview_json else fetch(args.skip, args.n))
    result = summarize(rows, args.skip, args.n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items()
                      if k != 'document_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()

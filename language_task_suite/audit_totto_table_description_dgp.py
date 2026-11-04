"""Full-release yield, token alignment, and form audit for ToTTo adapter."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import zipfile

from tokenizers import Tokenizer

from totto_table_description_dgp import check_visible, extract


TRAIN = 'totto_data/totto_train_data.jsonl'
DEFAULT_TOKENIZER = (Path.home() / '.cache/huggingface/hub/models--gpt2/'
                     'snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/'
                     'tokenizer.json')
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def form(texts):
    words = [[match.group().lower() for match in WORD.finditer(text)]
             for text in texts]
    unigram = Counter(word for doc in words for word in doc)
    trigrams = Counter(gram for doc in words
                       for gram in zip(doc, doc[1:], doc[2:]))
    doc_frequency = Counter()
    for doc in words:
        doc_frequency.update(set(zip(doc, doc[1:], doc[2:])))
    total_words = sum(unigram.values())
    total_trigrams = sum(trigrams.values())
    return {'n': len(texts), 'word_types': len(unigram),
            'mean_words': total_words / len(texts),
            'top10_word_share': sum(n for _, n in unigram.most_common(10))
                                / total_words,
            'repeated_across_docs_trigram_occurrence_share': sum(
                n for gram, n in trigrams.items() if doc_frequency[gram] > 1
            ) / total_trigrams}


def target_token_ids(tokenizer, example, site):
    encoding = tokenizer.encode(example.text)
    lo, hi = site.span
    ids = [i for i, (a, b) in enumerate(encoding.offsets)
           if a < hi and b > lo]
    if not ids:
        return None, len(encoding.ids)
    left = example.text[encoding.offsets[ids[0]][0]:lo]
    right = example.text[hi:encoding.offsets[ids[-1]][1]]
    if left.strip() or right.strip():
        return None, len(encoding.ids)
    return ids, len(encoding.ids)


def audit(source, *, seed=74801, panel_size=512):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    counts = Counter()
    column_pairs = Counter()
    pages = set()
    all_sentences, selected_sentences, examples = [], [], []
    rng = random.Random(seed)
    with zipfile.ZipFile(source) as archive, archive.open(TRAIN) as stream:
        for line in stream:
            row = json.loads(line)
            counts['records'] += 1
            annotations = row.get('sentence_annotations') or []
            if annotations:
                sentence = annotations[0].get('final_sentence', '').strip()
                if sentence:
                    counts['all_sentence_draws'] += 1
                    j = rng.randrange(counts['all_sentence_draws'])
                    if len(all_sentences) < panel_size:
                        all_sentences.append(sentence)
                    elif j < panel_size:
                        all_sentences[j] = sentence
            example = extract(row)
            if example is None:
                continue
            counts['two_exact_sites'] += 1
            check_visible(example)
            pages.add(row.get('table_webpage_url') or row['table_page_title'])
            counts['sites'] += len(example.sites)
            column_pairs[tuple(sorted(site.column.casefold()
                                      for site in example.sites[:2]))] += 1
            usable = []
            lengths = []
            for site in example.sites:
                ids, length = target_token_ids(tokenizer, example, site)
                usable.append(ids is not None and ids[-1] < 256)
                lengths.append(length)
                counts['aligned_sites'] += ids is not None
                counts['sites_under_256'] += bool(ids and ids[-1] < 256)
            counts['all_sites_under_256'] += all(usable)
            counts['full_document_under_256'] += max(lengths) <= 256
            selected_sentence = example.text.split('Description: ', 1)[1]
            j = rng.randrange(counts['two_exact_sites'])
            if len(selected_sentences) < panel_size:
                selected_sentences.append(selected_sentence)
            elif j < panel_size:
                selected_sentences[j] = selected_sentence
            if len(examples) < 12:
                examples.append(example)
    result = {
        'source': 'ToTTo official train JSONL in local archive',
        'seed': seed, 'counts': dict(counts),
        'distinct_pages': len(pages),
        'column_pairs': [[list(pair), n]
                          for pair, n in column_pairs.most_common(30)],
        'selected_form_n': len(selected_sentences),
        'selected_sentence_form': form(selected_sentences),
        'unselected_sentence_form': form(all_sentences),
        'caveat': 'Exact highlighted value mentions and visible-table '
                  'checks do not prove every sentence claim is supported '
                  'by its table or that an arbitrary value edit is safe. '
                  'This is a one-sentence grounded-data component, not '
                  'a whole-document rule oracle.'}
    return result, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=74801)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result, examples = audit(args.source, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'], flush=True)
    for example in examples[:8]:
        print(example.text[:650].replace('\n', ' / '), flush=True)


if __name__ == '__main__':
    main()

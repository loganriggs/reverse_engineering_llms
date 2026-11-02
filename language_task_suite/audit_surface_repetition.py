"""Compare within-window lexical repetition in synthetic and FineWeb text.

This is a surface diagnostic, not a semantic coreference annotation.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from surface_mixture_v2 import render
from surface_mixture_v3 import SurfaceMixtureV3


STOPWORDS = set('a an and are as at be been but by for from had has have he her him his '
                'i in is it its of on or our she that the their them there they this to '
                'was were when which while who with you your'.split())
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def score(texts, word_prefix=None):
    rows = []
    for text in texts:
        words = [x.lower() for x in WORD.findall(text)]
        if word_prefix is not None:
            words = words[:word_prefix]
        content = [x for x in words if x not in STOPWORDS]
        def repeated(tokens):
            seen = set()
            n = 0
            for token in tokens:
                n += token in seen
                seen.add(token)
            return n/len(tokens) if tokens else 0.0
        rows.append(dict(n_words=len(words), n_content=len(content),
                         repeated_word_fraction=repeated(words),
                         repeated_content_fraction=repeated(content),
                         unique_content=len(set(content))))
    return dict(n_texts=len(rows), **{
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]})


def form_cues(texts):
    patterns = {'newline': r'\n', 'question_mark': r'\?',
                'colon': r':', 'quote': r'["“”]',
                'parenthesis': r'[()]', 'digit': r'\d'}
    return {name: float(np.mean([bool(re.search(pattern, text))
                                 for text in texts]))
            for name, pattern in patterns.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=480)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--output', default='language_task_suite/surface_repetition_audit.json')
    a = p.parse_args()
    gen = SurfaceMixtureV3(seed=12001)
    ids, _, _ = gen.sample_batch(a.n, a.length)
    synthetic = [render(row) for row in ids]
    rows = torch.load(DEFAULT_ROWS, map_location='cpu', weights_only=True)
    assert a.n <= len(rows)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    natural = [tokenizer.decode(row[:a.length].tolist(),
                                skip_special_tokens=False)
               for row in rows[:a.n]]
    common_words = 40
    eligible = [i for i in range(a.n)
                if len(WORD.findall(synthetic[i])) >= common_words
                and len(WORD.findall(natural[i])) >= common_words]
    output = dict(config=vars(a), synthetic=score(synthetic),
                  fineweb=score(natural),
                  form_cues=dict(synthetic=form_cues(synthetic),
                                 fineweb=form_cues(natural)),
                  word_matched=dict(words_per_text=common_words,
                                    n_texts=len(eligible),
                                    synthetic=score([synthetic[i] for i in eligible],
                                                    common_words),
                                    fineweb=score([natural[i] for i in eligible],
                                                  common_words)),
                  caveat='Exact case-insensitive word repetition in truncated '
                         'token windows is not coreference or induction. The '
                         'synthetic and natural streams use different tokenizers '
                         'and the stopword list is a simple fixed heuristic.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

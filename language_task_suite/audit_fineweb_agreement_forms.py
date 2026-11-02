"""Count narrow agreement-like surface forms in frozen FineWeb windows.

These regexes are surface proxies, not syntactic parses or evidence that the
intervening noun controls (or fails to control) the finite verb.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER


PATTERNS = {
    'v3_pp_near_beside': (r'\b(?:the|a|an)\s+\w+\s+'
                          r'(?:near|beside)\s+(?:the|a|an)\s+\w+\s+'
                          r'(?:is|are|was|were)\b'),
    'broad_pp': (r'\b(?:the|a|an)\s+\w+\s+'
                 r'(?:of|with|on|in|by|for|near|beside)\s+'
                 r'(?:the|a|an)\s+\w+\s+(?:is|are|was|were)\b'),
    'v3_relative_that': (r'\b(?:the|a|an)\s+\w+\s+that\s+'
                         r'(?:the|a|an)\s+\w+\s+\w+\s+'
                         r'(?:is|are|was|were)\b'),
    'broad_relative': (r'\b(?:the|a|an)\s+\w+\s+'
                       r'(?:that|who|which)\s+(?:the|a|an)\s+'
                       r'\w+\s+\w+\s+(?:is|are|was|were)\b'),
}


def audit(rows_path, tokenizer_path):
    rows = torch.load(rows_path, map_location='cpu', weights_only=True)
    assert rows.ndim == 2 and rows.shape[1] == 513
    ids = rows.numpy()
    tok = Tokenizer.from_file(str(tokenizer_path))
    texts = [tok.decode(row.tolist(), skip_special_tokens=False)
             for row in ids]
    for i in (0, len(texts)//2, len(texts)-1):
        assert tok.encode(texts[i], add_special_tokens=False).ids == ids[i].tolist()
    results = {}
    for name, pattern in PATTERNS.items():
        hits = [(i, m.group(0)) for i, text in enumerate(texts)
                for m in re.finditer(pattern, text, re.IGNORECASE)]
        results[name] = dict(occurrences=len(hits),
                             windows=len({i for i, _ in hits}),
                             examples=[dict(window=i, text=text)
                                       for i, text in hits[:12]])
    return dict(source=str(rows_path), n_windows=len(texts),
                tokens_per_window=ids.shape[1],
                source_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                patterns=results,
                caveat='Case-insensitive regex surface cues only. These do not '
                       'establish grammatical agreement, structural ambiguity, '
                       'or natural-language frequencies outside these panels.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=Path, nargs='+', default=[DEFAULT_ROWS])
    p.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = [audit(rows, a.tokenizer) for rows in a.rows]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for panel in result:
        print(Path(panel['source']).name,
              {name: row['occurrences']
               for name, row in panel['patterns'].items()})

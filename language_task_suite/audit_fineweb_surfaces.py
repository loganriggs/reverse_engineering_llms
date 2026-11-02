"""Read a frozen local FineWeb token panel and count *surface* task cues.

These regexes are deliberately weak evidence about natural text frequency,
not linguistic labels or proof that a proposed grammar is correct.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer


HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
DEFAULT_ROWS = (CODING / 'reverse_engineer/tensor_language/'
                'basis_aligned/bilinear_quotient/.rowcache/fineweb_n480_skip80.pt')
DEFAULT_TOKENIZER = (Path.home() / '.cache/huggingface/hub/models--gpt2/'
                     'snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/'
                     'tokenizer.json')

PATTERNS = {
    'parenthesis_pair': r'\([^\n)]{1,160}\)',
    'square_bracket_pair': r'\[[^\n\]]{1,160}\]',
    'double_quote_pair': r'"[^"\n]{2,160}"|“[^”\n]{2,160}”',
    'numbered_list_line': r'(?m)^\s{0,5}\d{1,3}[.)]\s+\w',
    'bullet_list_line': r'(?m)^\s{0,5}[-*•]\s+\w',
    'weekday_name': r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
    'month_name': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',
    'if_then': r'\bif\b[^.!?\n]{0,180}\bthen\b',
    'relative_marker': r'\b(?:who|which|that)\b',
    'pronoun': r'\b(?:he|she|they|him|her|them|his|their)\b',
    'state_change_verb': r'\b(?:put|moved|placed|removed|stored|transferred)\b',
    'comparative': r'\b(?:more|less|older|younger|larger|smaller)\s+than\b',
    'iso_date': r'\b\d{4}-\d{2}-\d{2}\b',
    'question_mark': r'\?',
    'colon': r':',
}


def audit(rows_path, tokenizer_path):
    rows = torch.load(rows_path, map_location='cpu', weights_only=True)
    assert rows.ndim == 2 and rows.shape[1] == 513
    ids = rows.numpy()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    texts = [tokenizer.decode(row.tolist(), skip_special_tokens=False)
             for row in ids]
    assert len(set(texts)) == len(texts)
    # Round-trip detects a tokenizer/version mismatch before interpreting text.
    for i in (0, len(texts)//2, len(texts)-1):
        assert tokenizer.encode(texts[i], add_special_tokens=False).ids == ids[i].tolist()
    counts = {}
    for name, expression in PATTERNS.items():
        regex = re.compile(expression, flags=re.IGNORECASE)
        matches = [list(regex.finditer(text)) for text in texts]
        counts[name] = dict(windows=sum(bool(m) for m in matches),
                            occurrences=sum(len(m) for m in matches),
                            window_fraction=sum(bool(m) for m in matches)/len(texts))
    result = dict(source='frozen local FineWeb GPT-2-token panel; surface cues only',
        rows=str(rows_path), tokenizer=str(tokenizer_path),
        n_windows=len(texts), tokens_per_window=ids.shape[1],
        source_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        total_characters=sum(map(len, texts)), patterns=counts,
        caveat='A token window is not a document; regex hits are not validated linguistic dependencies or causal labels. This audit calibrates form/frequency only.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=Path, default=DEFAULT_ROWS)
    p.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument('--output', type=Path,
                   default=HERE/'fineweb_surface_audit.json')
    args = p.parse_args()
    result = audit(args.rows, args.tokenizer)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for name, value in result['patterns'].items():
        print(name, value['windows'], '/', result['n_windows'])

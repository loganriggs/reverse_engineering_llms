"""Support/runtime audit for sampled recursive PCFG text under EarleyX.

The Berkeley grammar and converted EarleyX grammar are local inputs. Samples
and parser logs stay in a temporary directory; only aggregate JSON is saved.
"""

import argparse
import collections
import hashlib
import json
import math
import random
import subprocess
import tempfile
import time
from pathlib import Path

from berkeley_pcfg_sampler import read_berkeley_grammar, sample_sentence


def _read_surprisals(path):
    rows, row = [], []
    for line in path.read_text().splitlines():
        if line.startswith('#'):
            if row:
                rows.append(row)
                row = []
        elif line.strip():
            word, value = line.split()
            row.append((word, float(value)))
    if row:
        rows.append(row)
    return rows


def audit(grammar_base, earleyx_grammar, jar, lib_dir, n=30,
          seed=83001, normalize_apostrophes=True):
    rules, metadata = read_berkeley_grammar(grammar_base)
    rng = random.Random(seed)
    rows = []
    rejected = collections.Counter()
    for _ in range(n):
        words, reason = sample_sentence(rules, rng)
        if reason:
            rejected[reason] += 1
            continue
        if normalize_apostrophes:
            words = tuple(w.replace("'", '<apostrophe>') for w in words)
        rows.append(words)
    with tempfile.TemporaryDirectory(prefix='pcfg_prefix_audit_') as tmp:
        root = Path(tmp)
        (root/'sample.txt').write_text(
            '\n'.join(' '.join(row) for row in rows)+'\n')
        command = ['java', '-Xmx4g', '-classpath', f'{jar}:{lib_dir}/*',
                   'parser.Main', '-in', str(root/'sample.txt'),
                   '-grammar', str(earleyx_grammar),
                   '-out', str(root/'score'), '-verbose', '0',
                   '-thread', '1', '-root', 'ROOT_0']
        start = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True,
                                 timeout=300)
        elapsed = time.perf_counter()-start
        assert process.returncode == 0, process.stderr[-1000:]
        parsed = _read_surprisals(root/'score.surprisal')
    assert len(parsed) == len(rows)
    assert all(tuple(w for w, _ in got) == want
               for got, want in zip(parsed, rows))
    failures = collections.Counter()
    finite_tokens = 0
    total_tokens = 0
    for row in parsed:
        total_tokens += len(row)
        finite_tokens += sum(math.isfinite(value) for _, value in row)
        for word, value in row:
            if not math.isfinite(value):
                failures[word] += 1
                break
    base = Path(grammar_base)
    return dict(n_requested=n, n_parsed=len(parsed), seed=seed,
                normalized_apostrophes=normalize_apostrophes,
                rejected=dict(rejected), grammar_metadata=metadata,
                grammar_sha256=hashlib.sha256(
                    base.with_suffix('.grammar').read_bytes()).hexdigest(),
                lexicon_sha256=hashlib.sha256(
                    base.with_suffix('.lexicon').read_bytes()).hexdigest(),
                earleyx_grammar_sha256=hashlib.sha256(
                    Path(earleyx_grammar).read_bytes()).hexdigest(),
                finite_sentences=len(parsed)-sum(failures.values()),
                finite_tokens=finite_tokens, total_tokens=total_tokens,
                first_nonfinite_token=dict(failures),
                parser_wall_seconds=elapsed,
                caveat=('Finite sampled-target surprisals are not yet a '
                        'validated full next-token distribution for the '
                        'joint document grammar. The parser grammar must '
                        'match every sampling rule and token normalization.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--grammar-base', type=Path, required=True)
    p.add_argument('--earleyx-grammar', type=Path, required=True)
    p.add_argument('--jar', type=Path, required=True)
    p.add_argument('--lib-dir', type=Path, required=True)
    p.add_argument('--n', type=int, default=30)
    p.add_argument('--seed', type=int, default=83001)
    p.add_argument('--raw-apostrophes', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.grammar_base, args.earleyx_grammar,
                   args.jar, args.lib_dir, args.n, args.seed,
                   not args.raw_apostrophes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print({k: result[k] for k in ('n_parsed', 'finite_sentences',
                                  'finite_tokens', 'total_tokens',
                                  'first_nonfinite_token',
                                  'parser_wall_seconds')})

"""Resumable FineWeb preview fetch; source prose stays in local artifacts.

Writes one JSONL row at a time, verifies contiguous offsets on resume, then
atomically exports the array format expected by existing cache builders.
Only the aggregate/hash manifest belongs in Git.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import islice
import json
import os
from pathlib import Path

from fetch_fineweb_long_panel import manifest


def _load_existing(path, skip, n):
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                raise ValueError('Incomplete final JSONL row; repair locally')
            row = json.loads(line)
            if row['index'] != skip+len(rows):
                raise ValueError('Noncontiguous resume offset')
            rows.append(row)
    if len(rows) > n:
        raise ValueError('Resume file exceeds requested n')
    return rows


def fetch(skip, n, preview_chars, jsonl, preview_json, manifest_json):
    from datasets import load_dataset

    jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_existing(jsonl, skip, n)
    done = len(rows)
    print('resuming', done, 'of', n, flush=True)
    if done < n:
        stream = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                              split='train', streaming=True)
        with jsonl.open('a') as output:
            for index, row in enumerate(islice(iter(stream), skip+done,
                                                skip+n), skip+done):
                text = row['text']
                record = {'index': index, 'url': row['url'],
                          'full_length': len(text),
                          'sha256': hashlib.sha256(text.encode()).hexdigest(),
                          'preview': text[:preview_chars]}
                output.write(json.dumps(record, ensure_ascii=False)+'\n')
                rows.append(record)
                if len(rows) % 1000 == 0:
                    output.flush()
                    os.fsync(output.fileno())
                    print('fetched', len(rows), 'of', n, flush=True)
            output.flush()
            os.fsync(output.fileno())
    if len(rows) != n:
        raise RuntimeError(f'Only fetched {len(rows)} of {n} source rows')
    preview_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = preview_json.with_suffix('.tmp')
    tmp.write_text(json.dumps(rows, ensure_ascii=False))
    tmp.replace(preview_json)
    report = manifest(rows, skip, n, preview_chars)
    report['resumable_jsonl_local'] = str(jsonl)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    temp = manifest_json.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2)+'\n')
    temp.replace(manifest_json)
    print('complete', n, 'sources', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, required=True)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--preview-chars', type=int, default=2500)
    p.add_argument('--jsonl', type=Path, required=True)
    p.add_argument('--preview-json', type=Path, required=True)
    p.add_argument('--manifest-json', type=Path, required=True)
    a = p.parse_args()
    fetch(a.skip, a.n, a.preview_chars, a.jsonl,
          a.preview_json, a.manifest_json)
    # The streaming datasets reader can leave non-daemon threads alive after
    # all rows and outputs have been flushed and verified.
    os._exit(0)


if __name__ == '__main__':
    main()

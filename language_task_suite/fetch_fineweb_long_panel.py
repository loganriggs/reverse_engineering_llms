"""Fetch an independent long-preview FineWeb development panel.

The preview JSON contains copyrighted source text and belongs in a temporary
local path; only the hash manifest should be committed.
"""

import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path
from urllib.parse import urlparse


def fetch(skip, n, preview_chars):
    from datasets import load_dataset

    stream = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    rows = []
    for offset, row in enumerate(islice(stream, skip, skip+n), skip):
        text = row['text']
        rows.append(dict(index=offset, url=row['url'],
                         full_length=len(text),
                         sha256=hashlib.sha256(text.encode()).hexdigest(),
                         preview=text[:preview_chars]))
    assert len(rows) == n
    return rows


def manifest(rows, skip, n, preview_chars):
    assert [r['index'] for r in rows] == list(range(skip, skip+n))
    return dict(source='HuggingFaceFW/fineweb sample-10BT train',
                selection=f'{n} contiguous rows at offsets {skip}..{skip+n-1}',
                preview_characters=preview_chars,
                distinct_urls=len({r['url'] for r in rows}),
                distinct_hosts=len({urlparse(r['url']).hostname for r in rows}),
                at_least_2000_chars=sum(r['full_length'] >= 2000 for r in rows),
                at_least_4000_chars=sum(r['full_length'] >= 4000 for r in rows),
                hashes=[dict(offset=r['index'], sha256=r['sha256'],
                             full_length=r['full_length']) for r in rows],
                caveat='Manifest omits URL and source text. Contiguous corpus '
                       'rows are a convenience sample, not a population draw.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=25000)
    p.add_argument('--n', type=int, default=48)
    p.add_argument('--preview-chars', type=int, default=4000)
    p.add_argument('--preview-json', type=Path, required=True)
    p.add_argument('--manifest-json', type=Path, required=True)
    args = p.parse_args()
    rows = fetch(args.skip, args.n, args.preview_chars)
    args.preview_json.write_text(json.dumps(rows))
    result = manifest(rows, args.skip, args.n, args.preview_chars)
    args.manifest_json.write_text(json.dumps(result, indent=2) + '\n')
    print({k: v for k, v in result.items() if k != 'hashes'}, flush=True)

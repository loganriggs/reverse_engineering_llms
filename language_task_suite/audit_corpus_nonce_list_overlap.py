"""Count exact source-document overlap in two corpus-grounded rule caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nonce-manifest', type=Path, required=True)
    p.add_argument('--list-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    nonce = json.loads(a.nonce_manifest.read_text())
    lists = json.loads(a.list_manifest.read_text())
    result = {'sources': [str(a.nonce_manifest), str(a.list_manifest)],
              'splits': {},
              'scope': 'Exact SHA-256 intersection of selected, aligned '
                       'source documents in the existing strict DGP caches. '
                       'This is not the prevalence of all possible acronym '
                       'and list patterns in general web text.'}
    for split in ('val', 'train', 'test'):
        nr, lr = nonce['splits'][split], lists['splits'][split]
        if nr['offsets'] != lr['offsets']:
            result['splits'][split] = {
                'nonce_offsets': nr['offsets'],
                'list_offsets': lr['offsets'],
                'comparison': 'not_comparable_different_source_ranges'}
            continue
        nh = {row['sha256'] for row in nr['selected_source_sha256']}
        lh = set(lr['source_sha256'])
        n_scanned = nr['offsets'][1] - nr['offsets'][0]
        assert nr['counts']['source_documents'] == n_scanned
        result['splits'][split] = {
            'source_documents_scanned': n_scanned,
            'nonce_selected': len(nh), 'list_selected': len(lh),
            'both_selected': len(nh & lh),
            'both_per_1000_scanned': 1000 * len(nh & lh) / n_scanned}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['splits'])


if __name__ == '__main__':
    main()

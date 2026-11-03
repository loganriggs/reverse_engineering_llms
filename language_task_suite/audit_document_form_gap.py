"""Compare generated word/adjacent-word form with a disjoint FineWeb preview.

The public corpus preview stays outside the repository. The output contains
only aggregate counts and source hashes. Adjacent-word pairs require only
whitespace between the two words, so punctuation/number-skipping artifacts
do not masquerade as collocations.
"""

import argparse
import collections
import json
import re
from pathlib import Path

from genre_document_v3 import sample_sequence as sample_v3
from genre_document_v4 import sample_sequence as sample_v4


WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def summarize(word_rows, pair_rows):
    words = collections.Counter(word for row in word_rows for word in row)
    pairs = collections.Counter(pair for row in pair_rows for pair in row)
    n_words = sum(words.values())
    n_pairs = sum(pairs.values())
    return dict(n_documents=len(word_rows), n_words=n_words,
                word_types=len(words),
                word_singletons=sum(v == 1 for v in words.values()),
                top10_word_share=sum(v for _, v in words.most_common(10))/n_words,
                n_adjacent_word_pairs=n_pairs,
                adjacent_pair_types=len(pairs),
                adjacent_pair_singletons=sum(v == 1 for v in pairs.values()),
                top10_adjacent_pair_share=(
                    sum(v for _, v in pairs.most_common(10))/n_pairs),
                top10_adjacent_pairs=[[list(k), v] for k, v in pairs.most_common(10)])


def real_rows(rows, manifest):
    assert len(rows) == len(manifest['hashes']) == 48
    assert [row['index'] for row in rows] == list(range(25000, 25048))
    assert [(row['index'], row['sha256'], row['full_length']) for row in rows] == [
        (row['offset'], row['sha256'], row['full_length'])
        for row in manifest['hashes']]
    word_rows, pair_rows = [], []
    for row in rows:
        text = row['preview'][:1200]
        matches = list(WORDS.finditer(text))[:100]
        if len(matches) < 100:
            continue
        word_rows.append([m.group().lower() for m in matches])
        pair_rows.append([(a.group().lower(), b.group().lower())
                          for a, b in zip(matches, matches[1:])
                          if text[a.end():b.start()].isspace()])
    return word_rows, pair_rows


def generated_rows(n, regime, seed, sampler):
    word_rows, pair_rows, chosen = [], [], []
    candidate = 0
    while len(word_rows) < n:
        assert candidate < 8*n, 'too few eligible generated documents'
        doc_seed = seed+candidate
        candidate += 1
        words, _, _, _ = sampler(doc_seed, 256, regime)
        positions = [i for i, word in enumerate(words) if word.isalpha()][:100]
        if len(positions) < 100:
            continue
        word_rows.append([words[i].lower() for i in positions])
        pair_rows.append([(words[i].lower(), words[j].lower())
                          for i, j in zip(positions, positions[1:])
                          if j == i+1])
        chosen.append(doc_seed)
    return word_rows, pair_rows, chosen, candidate


def audit(preview_path, manifest_path, seed=58001):
    rows = json.loads(preview_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    natural_words, natural_pairs = real_rows(rows, manifest)
    n = len(natural_words)
    assert n >= 30
    out = dict(source='FineWeb sample-10BT offsets 25000..25047, first1200 '
                      'characters, 100 alphabetic words when eligible',
               panel_sha256=[row['sha256'] for row in rows],
               real_documents_scanned=len(rows),
               real_eligible=n,
               natural=summarize(natural_words, natural_pairs),
               generated={},
               caveat='First-100-word form comparison of a convenience panel, '
                      'not a semantic task prevalence or a population estimate. '
                      'Generators use 256-token streams, FineWeb 1200-character '
                      'previews. Whitespace-adjacent pair denominators differ.')
    for version, sampler in (('v3', sample_v3), ('v4', sample_v4)):
        out['generated'][version] = {}
        for regime in ('task_rich', 'reference'):
            words, pairs, seeds, scanned = generated_rows(n, regime, seed,
                                                         sampler)
            out['generated'][version][regime] = dict(
                summarize(words, pairs), first_seed=seed,
                candidates_scanned=scanned, chosen_seeds=seeds)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--preview-json', type=Path, required=True)
    p.add_argument('--manifest-json', type=Path,
                   default=Path(__file__).parent/'results'/
                   'fineweb_long_panel_25000_n48_manifest.json')
    p.add_argument('--seed', type=int, default=58001)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.preview_json, args.manifest_json, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print({key: {k: v for k, v in result[key].items()
                 if k not in ('top10_adjacent_pairs', 'chosen_seeds')}
           for key in ('natural',)}, flush=True)
    print({version: {regime: {k: v for k, v in row.items()
                              if k not in ('top10_adjacent_pairs',
                                           'chosen_seeds')}
                     for regime, row in regimes.items()}
           for version, regimes in result['generated'].items()}, flush=True)

"""Form-only screen of a locally induced Berkeley PCFG against FineWeb.

No generated text or source preview is saved. This audit does not certify an
exact next-token oracle or semantically coherent documents.
"""

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from audit_document_form_gap import WORDS, real_rows, summarize
from berkeley_pcfg_sampler import read_berkeley_grammar, sample_sentence


def audit(base_path, preview_path, manifest_path, seed=58001):
    rows = json.loads(preview_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    natural_words, natural_pairs = real_rows(rows, manifest)
    n_docs = len(natural_words)
    rng = random.Random(seed)
    rules, metadata = read_berkeley_grammar(base_path)
    word_rows, pair_rows = [], []
    sentence_lengths = []
    failures = collections.Counter()
    for _ in range(n_docs):
        tokens = []
        while sum(WORDS.fullmatch(w) is not None for w in tokens) < 100:
            sentence, reason = sample_sentence(rules, rng)
            if reason:
                failures[reason] += 1
                assert sum(failures.values()) < 10_000
                continue
            sentence_lengths.append(len(sentence))
            tokens.extend(sentence)
        positions = [i for i, word in enumerate(tokens)
                     if WORDS.fullmatch(word)][:100]
        word_rows.append([tokens[i].lower() for i in positions])
        pair_rows.append([(tokens[i].lower(), tokens[j].lower())
                          for i, j in zip(positions, positions[1:])
                          if j == i+1])
    base = Path(base_path)
    return dict(source='Locally induced Berkeley PCFG from paper treebank; '
                       'form-only diagnostic', seed=seed,
                input_grammar_sha256=hashlib.sha256(
                    base.with_suffix('.grammar').read_bytes()).hexdigest(),
                input_lexicon_sha256=hashlib.sha256(
                    base.with_suffix('.lexicon').read_bytes()).hexdigest(),
                grammar_metadata=metadata,
                sample_failures=dict(failures),
                generated_sentences=len(sentence_lengths),
                mean_generated_sentence_tokens=(
                    sum(sentence_lengths)/len(sentence_lengths)),
                natural=summarize(natural_words, natural_pairs),
                generated=summarize(word_rows, pair_rows),
                caveat=('Recursive grammar sampled with explicit safeguards; '
                        'a failed expansion is rejected. This audit tests '
                        'surface form only and must not be scored against '
                        'the unconditioned PCFG oracle if failures occur. '
                        'The sentences are concatenated without a document '
                        'planner or world semantics.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--grammar-base', type=Path, required=True)
    p.add_argument('--preview-json', type=Path, required=True)
    p.add_argument('--manifest-json', type=Path,
                   default=Path(__file__).parent/'results'/
                   'fineweb_long_panel_25000_n48_manifest.json')
    p.add_argument('--seed', type=int, default=58001)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.grammar_base, args.preview_json,
                   args.manifest_json, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for key in ('grammar_metadata', 'sample_failures', 'generated_sentences',
                'mean_generated_sentence_tokens'):
        print(key, result[key])
    for key in ('natural', 'generated'):
        print(key, {k: v for k, v in result[key].items()
                    if k != 'top10_adjacent_pairs'})

"""Matched GPT-2 form-likelihood diagnostic; source text stays local."""

import argparse
import hashlib
import json
import random
import re
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_document_form_gap import WORDS, real_rows
from berkeley_pcfg_sampler import read_berkeley_grammar, sample_sentence


TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[.,!?;:()\[\]\"']")


def _normalized(text):
    return ' '.join(TOKEN.findall(text))


def _generated_docs(rules, n, seed):
    rng = random.Random(seed)
    docs = []
    failures = 0
    for _ in range(n):
        tokens = []
        while sum(bool(WORDS.fullmatch(t)) for t in tokens) < 100:
            sentence, reason = sample_sentence(rules, rng)
            if reason:
                failures += 1
                continue
            tokens.extend(sentence)
        last = [i for i, t in enumerate(tokens)
                if WORDS.fullmatch(t)][99]
        docs.append(_normalized(' '.join(tokens[:last+1])))
    return docs, failures


def _control_docs(sampler, n, seed):
    docs = []
    candidate = 0
    while len(docs) < n:
        assert candidate < 8*n
        tokens, _, _, _ = sampler(seed+candidate, 256, 'reference')
        candidate += 1
        positions = [i for i, token in enumerate(tokens)
                     if WORDS.fullmatch(token)]
        if len(positions) >= 100:
            docs.append(_normalized(' '.join(tokens[:positions[99]+1])))
    return docs


@torch.no_grad()
def _score(texts, tokenizer, model):
    rows = []
    total_bytes = sum(len(text.encode('utf-8')) for text in texts)
    for text in texts:
        ids = tokenizer(text, return_tensors='pt')['input_ids']
        logits = model(ids).logits
        losses = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                 ids[:, 1:].reshape(-1), reduction='none')
        rows.append((losses.sum().item(), losses.numel()))
    return dict(documents=len(rows),
                token_nll=sum(v for v, _ in rows)/sum(n for _, n in rows),
                nll_per_utf8_byte=sum(v for v, _ in rows)/total_bytes,
                median_document_token_nll=statistics.median(v/n for v,n in rows),
                bpe_target_tokens=sum(n for _, n in rows),
                utf8_bytes=total_bytes)


def audit(grammar_base, preview_path, manifest_path, seed=58001,
          include_controls=False):
    rows = json.loads(preview_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    natural_words, _ = real_rows(rows, manifest)
    natural = []
    for row in rows:
        preview = row['preview'][:1200]
        matches = list(WORDS.finditer(preview))[:100]
        if len(matches) == 100:
            natural.append(_normalized(preview[:matches[-1].end()]))
    assert len(natural) == len(natural_words) == 43
    rules, _ = read_berkeley_grammar(grammar_base)
    generated, failures = _generated_docs(rules, len(natural), seed)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2', local_files_only=True).eval().to('cpu')
    base = Path(grammar_base)
    result = dict(model='gpt2', device='cpu', seed=seed,
                normalization='shared regex token extraction, joined with spaces',
                source='held-out FineWeb panel 25000..25047',
                grammar_sha256=hashlib.sha256(
                    base.with_suffix('.grammar').read_bytes()).hexdigest(),
                generated_failed_expansions=failures,
                natural=_score(natural, tokenizer, model),
                induced_pcfg=_score(generated, tokenizer, model),
                caveat=('GPT-2 NLL is a diagnostic of natural-text alignment, '
                        'not an oracle or a measure of semantic correctness. '
                        'The same whitespace normalization is applied to '
                        'both sides; the source text is not saved.'))
    if include_controls:
        from corpus_backed_document_v5 import sample_sequence as sample_v5
        from corpus_markov_document_v6 import sample_sequence as sample_v6
        from genre_document_v7 import sample_sequence as sample_v7
        from genre_document_v8 import sample_sequence as sample_v8
        samplers = dict(v5=sample_v5, v6=sample_v6,
                        v7=sample_v7, v8=sample_v8)
        result['reference_controls'] = {
            name: _score(_control_docs(sampler, len(natural), seed),
                         tokenizer, model)
            for name, sampler in samplers.items()}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--grammar-base', type=Path, required=True)
    p.add_argument('--preview-json', type=Path, required=True)
    p.add_argument('--manifest-json', type=Path,
                   default=Path(__file__).parent/'results'/
                   'fineweb_long_panel_25000_n48_manifest.json')
    p.add_argument('--seed', type=int, default=58001)
    p.add_argument('--include-controls', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    q = audit(a.grammar_base, a.preview_json, a.manifest_json, a.seed,
              a.include_controls)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(q, indent=2)+'\n')
    print(q['natural'], q['induced_pcfg'])
    if a.include_controls:
        print(q['reference_controls'])

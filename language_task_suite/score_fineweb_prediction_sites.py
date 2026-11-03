"""Exploratory GPT-2 prefix/semantic-edit scores at annotated FineWeb sites.

This measures a model's sensitivity, not the ground-truth next-token rule.
Only source hashes, offsets, target tokens and scores are written.
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ALTERNATIVES = {23006: 'He', 23014: 'fourth floor', 23019: '11B',
                23033: '6-4', 23037: 'Jones', 23038: 'later'}
EDITS = {
    23006: [('A young female', 'A young male')],
    23014: [('third floor', 'fourth floor')],
    23019: [('Category: 11B', 'Category: 9C')],
    23033: [('SWB 5 – PAW 4', 'SWB 6 – PAW 4'),
            ('Pawtucket 5-4', 'Pawtucket 6-4')],
    23037: [('Mathis', 'Jones')],
    23038: [('Daylight Saving Time ended',
             'Daylight Saving Time started')],
    23043: [('Water resistance tested at 50M',
             'Water resistance tested at 20M')],
}


def suffix_words(text, n):
    matches = list(re.finditer(r'\S+', text))
    return text[matches[-n].start():] if len(matches) >= n else text


def edit_prefix(prefix, edits):
    out = prefix
    for old, new in edits:
        assert old in out, old
        out = out.replace(old, new)
    return out


@torch.no_grad()
def score(forward, tokenizer, device, prefix, target):
    full = prefix + target
    encoded = tokenizer(full, return_offsets_mapping=True)
    ids = encoded['input_ids']
    offsets = encoded['offset_mapping']
    start, end = len(prefix), len(full)
    selected = [i for i, (a, b) in enumerate(offsets)
                if b > start and a < end]
    assert selected and min(selected) >= 1
    assert max(selected) == len(ids)-1
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    logits = forward(tensor)[0]
    logp = torch.log_softmax(logits[:-1], dim=-1)
    return dict(logprob=sum(logp[i-1, ids[i]].item() for i in selected),
                context_tokens=min(selected), target_tokens=len(selected))


def run(preview_json, ledger_json, model_name, device,
        tensor_language_root=None):
    rows = {r['index']: r for r in json.loads(Path(preview_json).read_text())}
    ledger = json.loads(Path(ledger_json).read_text())
    tokenizer = AutoTokenizer.from_pretrained('gpt2' if model_name == 'bilin18'
                                              else model_name)
    if model_name == 'bilin18':
        import sys
        root = Path(tensor_language_root) if tensor_language_root else (
            Path(__file__).resolve().parents[2] / 'reverse_engineer/tensor_language')
        if not (root / 'basis_aligned/qk_mdl/tier2_model.py').exists():
            raise FileNotFoundError(f'pass --tensor-language-root for {root}')
        sys.path[:0] = [str(root), str(root / 'basis_aligned/qk_mdl')]
        from tier2_model import load_elriggs, reference_forward
        model, _ = load_elriggs('bilin18', device=device, dtype=torch.float32)
        forward = lambda ids: reference_forward(model, ids)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        forward = lambda ids: model(ids).logits
    results = []
    for site in ledger['sites']:
        row = rows[site['source_row']]
        assert row['sha256'] == site['source_sha256']
        text = row['preview']
        start, end = site['start_char'], site['end_char']
        target = site['target']
        assert text[start:end] == target
        prefix = text[:start]
        alt = ALTERNATIVES.get(site['source_row'])
        scores = {name:score(forward, tokenizer, device, context, target)
                  for name, context in [('full',prefix),
                                        ('last_100_words',suffix_words(prefix,100)),
                                        ('last_40_words',suffix_words(prefix,40))]}
        edited = EDITS.get(site['source_row'])
        if edited:
            edited_prefix = edit_prefix(prefix, edited)
            scores['edited_full'] = score(forward, tokenizer, device,
                                           edited_prefix, target)
        if alt:
            scores['alternative_full'] = score(forward, tokenizer, device,
                                                 prefix, alt)
            if edited:
                scores['alternative_edited'] = score(forward, tokenizer, device,
                                                       edited_prefix, alt)
        result = dict(source_row=site['source_row'],
                      source_sha256=site['source_sha256'],
                      family=site['family'], predictability=site['predictability'],
                      target=target, alternative=alt,
                      edit_kind=site['disambiguating_edit'] if edited else None,
                      scores=scores)
        results.append(result)
        print(site['source_row'],site['family'],
              round(scores['full']['logprob'],3),
              round(scores['last_40_words']['logprob'],3),flush=True)
    return dict(model=model_name, device=device, source='FineWeb sample-10BT',
                scope='8 purposively selected sites; exploratory model sensitivity',
                caveat=('Truncation changes the prefix and source text contains answer '
                        'leakage at some sites. A pretrained model is not a causal '
                        'oracle for natural language. No population estimate.'),
                sites=results)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preview-json',required=True)
    parser.add_argument('--ledger-json',required=True)
    parser.add_argument('--model',default='gpt2')
    parser.add_argument('--tensor-language-root')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=run(args.preview_json,args.ledger_json,args.model,args.device,
               args.tensor_language_root)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')

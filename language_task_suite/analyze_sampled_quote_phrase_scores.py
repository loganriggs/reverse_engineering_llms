"""Score both complete noun-phrase candidates under both source assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from train_corpus_delimiter_bilinear import _model


PAD = 50256


def candidate(arm, i, tokenizer):
    """Read the phrase continuation from cached y, stopping at closing quote."""
    start = int(arm['site'][i])
    out = []
    for token in arm['y'][i, start:].tolist():
        if token == -100:
            break
        if '"' in tokenizer.id_to_token(token):
            break
        out.append(token)
    if not out or out[0] != int(arm['target'][i]):
        raise ValueError('Malformed phrase continuation')
    return out


@torch.no_grad()
def batch_scores(model, prompts, device):
    """Mean token log-probability of each (prefix, candidate) pair."""
    max_length = max(len(prefix) + len(phrase)-1
                     for prefix, phrase in prompts)
    x = torch.full((len(prompts), max_length), PAD,
                   dtype=torch.long, device=device)
    for i, (prefix, phrase) in enumerate(prompts):
        ids = prefix + phrase[:-1]
        x[i, :len(ids)] = torch.tensor(ids, device=device)
    logits = model(x).float()
    logp = logits.log_softmax(-1)
    result = []
    for i, (prefix, phrase) in enumerate(prompts):
        positions = torch.arange(len(prefix)-1,
                                 len(prefix)+len(phrase)-1,
                                 device=device)
        values = logp[i, positions,
                      torch.tensor(phrase, device=device)]
        result.append(float(values.mean()))
    return result


def evaluate(model, data, tokenizer, device, limit=None, batch_pairs=8):
    a, b = data
    n = min(len(a['x']), limit or len(a['x']))
    paired = direction = 0
    margins = []
    for start in range(0, n, batch_pairs):
        prompts = []
        for i in range(start, min(n, start + batch_pairs)):
            phrase_a = candidate(a, i, tokenizer)
            phrase_b = candidate(b, i, tokenizer)
            prefix_a = a['x'][i, :int(a['site'][i])+1].tolist()
            prefix_b = b['x'][i, :int(b['site'][i])+1].tolist()
            prompts.extend(((prefix_a, phrase_a), (prefix_a, phrase_b),
                            (prefix_b, phrase_a), (prefix_b, phrase_b)))
        scores = []
        # Four equivalent-size batches bound the full-vocabulary logit buffer.
        for k in range(0, len(prompts), batch_pairs):
            scores.extend(batch_scores(model, prompts[k:k+batch_pairs], device))
        for k in range(0, len(scores), 4):
            aa, ab, ba, bb = scores[k:k+4]
            before, after = aa-ab, bb-ba
            paired += before > 0 and after > 0
            direction += before + after > 0
            margins.append(before + after)
    return {'pairs': n, 'paired_rank': paired/n,
            'source_swap_direction': direction/n,
            'mean_symmetric_margin_nats_per_token': sum(margins)/n,
            'scoring': 'Mean per-token autoregressive log probability of '
                       'each complete phrase. Positive symmetric margin '
                       'means that source assignment increases preference '
                       'for its own phrase in both arms jointly.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache = torch.load(args.cache, map_location='cpu', weights_only=True)
    saved = torch.load(args.state, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    model = _model(cache['vocab_size'], cache['n_ctx'], args.depth,
                   args.d_model, args.heads, args.device, 'rms')
    model.load_state_dict(saved['model'])
    model.eval()
    splits = {name: evaluate(model, data, tokenizer, args.device,
                             args.limit) for name, data in cache['pairs'].items()}
    out = {'checkpoint_step': saved['step'],
           'config': {'depth': args.depth, 'd_model': args.d_model,
                      'heads': args.heads, 'limit_per_split': args.limit},
           'splits': splits,
           'caveat': 'Per-token normalization compares phrases of different '
                     'lengths. Paired reversal controls a fixed candidate '
                     'prior but does not prove a unique neural circuit. '
                     'The selected noun chunk is sampled from multiple '
                     'valid phrases.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(args.depth, {k: round(v['paired_rank'], 3)
                       for k, v in splits.items()})


if __name__ == '__main__':
    main()

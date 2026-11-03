"""Causal source-to-query attention-edge screen with matched controls."""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import torch

from document_world import NAMES
from document_world_v5_clause_realizations import _canonical_event
from document_world_v8_discourse import _normalize_complete
from eval_document_world_v8_paired import collect, load_model


def _source_position(words, positions, source):
    if words[:2] == ['According', 'to']:
        offset = 2
    elif words[:1] == ['In'] and words[1].endswith("'s"):
        offset = 1
    elif words[:3] == ['A', 'notice', 'from']:
        offset = 3
    elif words[:4] == ['The', 'update', 'came', 'from']:
        offset = 4
    elif words[-3:-1] == ['according', 'to']:
        offset = len(words) - 1
    else:
        offset = 0
    assert words[offset] in (source, source + "'s"), (words, source)
    return positions[offset]


@lru_cache(maxsize=8192)
def event_sources(tokens):
    """Return visible report sources and their exact token positions."""
    events = []
    begin = 0
    for end, token in enumerate(tokens):
        if token != '.':
            continue
        positioned = [(index, word) for index, word in
                      enumerate(tokens[begin:end], start=begin)
                      if word != '\n']
        begin = end + 1
        words = [word for _, word in positioned]
        if not words:
            continue
        try:
            normalized = _normalize_complete(words)
            canonical = _canonical_event(normalized)
        except (AssertionError, IndexError, TypeError, ValueError):
            continue
        source = canonical[0]
        events.append((source, _source_position(
            words, [position for position, _ in positioned], source)))
    return tuple(events)


def key_index(doc, site, kind):
    query = site.index - 1
    if kind == 'self':
        return query
    if kind == 'source':
        plan = next(plan for recorded, plan in zip(
            doc.sites, doc.latent['recap_plan'], strict=True)
            if recorded.index == site.index)
        assert plan[1] == 'source_attribution'
        source, position = event_sources(doc.tokens)[plan[0]]
        assert source == site.answer and position < query
        return position
    else:
        assert kind == 'other_name'
        matches = [index for index, token in enumerate(doc.tokens[:query])
                   if token in NAMES and token != site.answer]
    return matches[-1] if matches else None


@torch.no_grad()
def probabilities(model, entries, vocab, n_ctx, device, batch_size,
                  edge=None):
    result = []
    for start in range(0, len(entries), batch_size):
        part = entries[start:start+batch_size]
        query = torch.tensor([site.index-1 for doc, site, target in part],
                             device=device)
        target = torch.tensor([vocab[target] for doc, site, target in part],
                              device=device)
        x = torch.full((len(part), n_ctx), vocab['<PAD>'],
                       dtype=torch.long, device=device)
        for index, (doc, _, _) in enumerate(part):
            ids = [vocab[token] for token in doc.tokens[:-1]]
            assert len(ids) <= n_ctx
            x[index, :len(ids)] = torch.tensor(ids, device=device)
        if edge is None:
            logits = model(x)
        else:
            layer_index, head_index, kind = edge
            layer = model.layers[layer_index]
            keys = [key_index(doc, site, kind) for doc, site, _ in part]
            assert all(key is not None and key <= site.index-1
                       for key, (_, site, _) in zip(keys, part, strict=True))
            key = torch.tensor(keys, device=device)
            original = layer.pattern

            def modified(activations):
                pattern = original(activations).clone()
                pattern[torch.arange(len(part), device=device), head_index,
                        query, key] = 0.
                return pattern

            layer.pattern = modified
            try:
                logits = model(x)
            finally:
                layer.pattern = original
        score = logits[torch.arange(len(part), device=device), query]
        result.extend(score.float().softmax(-1).gather(
            -1, target[:, None]).squeeze(-1).cpu().tolist())
    return result


def response(model, rows, vocab, n_ctx, device, batch_size, edge=None):
    before = [(doc, site, new) for doc, site, other, other_site, new in rows]
    after = [(other, other_site, new)
             for doc, site, other, other_site, new in rows]
    p0 = probabilities(model, before, vocab, n_ctx, device, batch_size, edge)
    p1 = probabilities(model, after, vocab, n_ctx, device, batch_size, edge)
    return sum(right-left for left, right in zip(p0, p1, strict=True)) / len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=349991)
    parser.add_argument('--streams', type=int, default=500)
    parser.add_argument('--max-cases', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, saved = load_model(args.checkpoint, args.device)
    kwargs = dict(saved['generator_kwargs'], source_delay_rate=0.)
    _, cases, skipped = collect(args.streams, args.seed, kwargs,
                                saved['n_ctx'], args.max_cases)
    rows = [row for row in cases['source_immediate']
            if all(key_index(doc, site, 'other_name') is not None
                   for doc, site in ((row[0], row[1]), (row[2], row[3])))]
    assert len(rows) >= 32
    vocab, n_ctx = saved['vocabulary'], saved['n_ctx']
    baseline = response(model, rows, vocab, n_ctx, args.device,
                        args.batch_size)
    measurements = []
    for layer_index, layer in enumerate(model.layers):
        if not hasattr(layer, 'pattern'):
            continue
        for head_index in range(layer.n_head):
            for kind in ('source', 'other_name', 'self'):
                ablated = response(model, rows, vocab, n_ctx, args.device,
                                   args.batch_size,
                                   (layer_index, head_index, kind))
                measurements.append(dict(layer=layer_index, head=head_index,
                                         edge_kind=kind,
                                         response=ablated,
                                         response_drop=baseline-ablated))
    output = dict(config={**vars(args), 'checkpoint': str(args.checkpoint),
                          'output': str(args.output)},
                  n_cases=len(rows), skipped=skipped, baseline=baseline,
                  edges=measurements,
                  caveat='One bilinear attention coefficient at the answer '
                         'query is zeroed. The source key is located in the '
                         'specific fact-bearing report indexed by the sampled '
                         'event plan, including possessive name tokens. This '
                         'ground-truth label selects the evaluation edge; '
                         'it is not an unsupervised discovery. The other-name '
                         'control is the last different person token. A '
                         'single-edge drop is causal for this intervention '
                         'but not a complete or unique path proof.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print('baseline', round(baseline, 3), 'top_drops', sorted(
        ((row['layer'], row['head'], row['edge_kind'],
          round(row['response_drop'], 3)) for row in measurements),
        key=lambda row: -row[3])[:10], flush=True)


if __name__ == '__main__':
    main()

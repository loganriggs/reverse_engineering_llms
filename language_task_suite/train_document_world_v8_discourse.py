"""Paired full-next-token bilinear pilots on V8 surface variation settings."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from document_world_v8_discourse import draw_varied, realize
from document_world import (COLORS, NAMES, OBJECTS as WORLD_OBJECTS,
                            PLACES, VALUES)
from document_world_v3_multisite import ALIASES
from surface_mixture_v2 import ADJECTIVES, EVENTS, OBJECTS as INCIDENTAL_OBJECTS
from train_document_world_v5_clause import DeepModel, SPECS, encode, evaluate


def sample_documents(n, seed, **kwargs):
    rng = random.Random(seed)
    return [draw_varied(rng, **kwargs) for _ in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--train-n', type=int, default=8000)
    parser.add_argument('--test-n', type=int, default=1000)
    parser.add_argument('--calibration-n', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--seed', type=int, default=34801)
    parser.add_argument('--syntax-entropies', default='0,1')
    parser.add_argument('--aside-rate', type=float, default=.3)
    parser.add_argument('--context-rate', type=float, default=.35)
    parser.add_argument('--source-delay-rate', type=float, default=0.)
    parser.add_argument('--extra-source-recap-rate', type=float, default=0.)
    parser.add_argument('--n-ctx', type=int, default=320)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert args.depth in (1, 2, 3, 4)
    settings = [float(value) for value in args.syntax_entropies.split(',')]
    assert len(settings) == 2 and all(0 <= value <= 1 for value in settings)
    common = dict(aside_rate=args.aside_rate, context_rate=args.context_rate,
                  source_delay_rate=args.source_delay_rate,
                  extra_source_recap_rate=args.extra_source_recap_rate,
                  paragraph_rate=.22, header_rate=.5)

    # Same latent worlds across settings. A union vocabulary and fixed context
    # width keep the architecture and parameter count matched.
    panels = {}
    for entropy in settings:
        kwargs = dict(common, syntax_entropy=entropy)
        panels[entropy] = dict(
            train=sample_documents(args.train_n, args.seed, **kwargs),
            test=sample_documents(args.test_n, args.seed + 100000, **kwargs))
    for name in ('train', 'test'):
        first = panels[settings[0]][name]
        second = panels[settings[1]][name]
        assert all(a.latent == b.latent for a, b in zip(first, second,
                                                         strict=True))
        assert all([(s.family, s.answer) for s in a.sites] ==
                   [(s.family, s.answer) for s in b.sites]
                   for a, b in zip(first, second, strict=True))
    vocab_words = {'<PAD>'}
    vocab_words.update(ADJECTIVES)
    vocab_words.update(EVENTS)
    vocab_words.update(INCIDENTAL_OBJECTS)
    vocab_words.update(WORLD_OBJECTS)
    vocab_words.update(COLORS)
    vocab_words.update(PLACES)
    vocab_words.update(NAMES)
    vocab_words.update(ALIASES)
    vocab_words.update(value for values in VALUES.values() for value in values)
    vocab_words.update(name + "'s" for name in NAMES)
    vocab_words.update(str(digit) for digit in range(10))
    for panel in panels.values():
        for docs in panel.values():
            vocab_words.update(word for doc in docs for word in doc.tokens)
    calibration = sample_documents(args.calibration_n, args.seed + 200000,
                                   **dict(common, syntax_entropy=1.,
                                          context_rate=1., aside_rate=1.,
                                          header_rate=1.))
    vocab_words.update(word for doc in calibration for word in doc.tokens)
    vocab = {word: index for index, word in enumerate(sorted(vocab_words))}
    max_length = max(len(doc.tokens) - 1 for panel in panels.values()
                     for docs in panel.values() for doc in docs)
    assert max_length <= args.n_ctx, (max_length, args.n_ctx)
    for panel in panels.values():
        for name in ('train', 'test'):
            panel[name + '_data'] = encode(panel[name], vocab, args.n_ctx)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for entropy in settings:
        panel = panels[entropy]
        torch.manual_seed(args.seed + args.depth)
        model = DeepModel(len(vocab), args.d_model, args.heads,
                          SPECS[f'block{args.depth}'], args.n_ctx,
                          norm='rms', residual='lerp',
                          attention='bilinear').to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rng = random.Random(args.seed + 300000 + args.depth)
        losses = []
        start = time.time()
        model.train()
        for step in range(args.steps):
            indexes = torch.tensor(rng.sample(range(args.train_n),
                                               args.batch_size))
            x, y = (tensor[indexes].to(args.device) for tensor in
                    panel['train_data'])
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                   ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss entropy={entropy} '
                                   f'step={step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
            if (step + 1) % 100 == 0:
                print('entropy', entropy, 'step', step + 1,
                      'loss', round(loss.item(), 4), flush=True)
        result = dict(config={**vars(args), 'output_dir': str(args.output_dir),
                              'syntax_entropy': entropy},
                      n_vocab=len(vocab), n_ctx=args.n_ctx,
                      max_document_length=max_length,
                      parameters=model.n_params,
                      train_last50=sum(losses[-50:]) / len(losses[-50:]),
                      test=evaluate(model, panel['test'], panel['test_data'],
                                    vocab, args.device),
                      seconds=round(time.time() - start, 2),
                      caveat='Known DGP rule labels are evaluation only; '
                             'surface realism remains poor. One model seed.')
        name = f'v8_syntax{entropy:g}_depth{args.depth}_seed{args.seed}'
        result_path = args.output_dir / f'{name}.json'
        checkpoint_path = args.output_dir / f'{name}.pt'
        result_path.write_text(json.dumps(result, indent=2) + '\n')
        torch.save(dict(state_dict={key: value.detach().cpu() for key, value in
                                    model.state_dict().items()},
                        vocabulary=vocab, n_ctx=args.n_ctx, depth=args.depth,
                        model_kwargs=dict(d_model=args.d_model,
                                          n_head=args.heads, norm='rms',
                                          residual='lerp',
                                          attention='bilinear'),
                        generator_kwargs=dict(common, syntax_entropy=entropy),
                        evaluation=result), checkpoint_path)
        print('saved', result_path, checkpoint_path, flush=True)


if __name__ == '__main__':
    main()

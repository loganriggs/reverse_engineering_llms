"""Attention L1/prune followed by strictly no-L1 recovery on V8 models."""

import argparse
import json
import time
from pathlib import Path

import torch

from sparsify_document_world_v5_clause import (
    enforce, global_prune, mask_stats, phase, prunable)
from train_document_world_v5_clause import DeepModel, SPECS, encode, evaluate
from train_document_world_v8_discourse import sample_documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--targets', default='.4,.6')
    parser.add_argument('--steps-per-round', type=int, default=80)
    parser.add_argument('--final-steps', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--l1', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=35441)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert saved['depth'] == 2
    config = saved['evaluation']['config']
    vocab, n_ctx = saved['vocabulary'], saved['n_ctx']
    kwargs = saved['generator_kwargs']
    train = sample_documents(config['train_n'], config['seed'], **kwargs)
    test = sample_documents(config['test_n'], config['seed']+100000, **kwargs)
    train_data = encode(train, vocab, n_ctx)
    test_data = encode(test, vocab, n_ctx)
    model = DeepModel(len(vocab), saved['model_kwargs']['d_model'],
                      saved['model_kwargs']['n_head'], SPECS['block2'],
                      n_ctx, norm=saved['model_kwargs']['norm'],
                      residual=saved['model_kwargs']['residual'],
                      attention=saved['model_kwargs']['attention']).to(args.device)
    model.load_state_dict(saved['state_dict'])
    params = prunable(model)
    assert len(params) == 12
    mask = {name: torch.ones_like(weight, dtype=torch.bool)
            for name, weight in params.items()}
    targets = [float(value) for value in args.targets.split(',')]
    assert targets == [.4, .6]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.time()

    def record(stage, train_loss=None):
        scores = evaluate(model, test, test_data, vocab, args.device)
        stats = mask_stats(mask)
        total = sum(weight.numel() for weight in model.parameters())
        row = dict(stage=stage, train_loss_last50=train_loss,
                   attention_mask=stats, total_params=total,
                   total_live=total-(stats['total']-stats['live']),
                   evaluation=scores, seconds=round(time.time()-start, 2))
        rows.append(row)
        payload = dict(state_dict={key: value.detach().cpu() for key, value
                                   in model.state_dict().items()},
                       mask={key: value.detach().cpu() for key, value
                             in mask.items()},
                       vocabulary=vocab, n_ctx=n_ctx, depth=2,
                       model_kwargs=saved['model_kwargs'],
                       generator_kwargs=kwargs,
                       evaluation=dict(config=config),
                       sparsity_config={**vars(args),
                                        'checkpoint': str(args.checkpoint),
                                        'output_dir': str(args.output_dir)},
                       stage_evaluation=row)
        torch.save(payload, args.output_dir / f'{stage}.pt')
        (args.output_dir / 'results.json').write_text(json.dumps(dict(
            config=payload['sparsity_config'], rows=rows,
            caveat='Attention q1/k1/q2/k2/v/o only. Embedding, '
                   'unembedding, MLP and norms remain dense. The final '
                   'recovery phase has zero L1 and an enforced mask; '
                   'causal rule edits must be evaluated separately.'),
            indent=2) + '\n')
        print(stage, 'attention_live', stats['live'],
              'source_accuracy', round(scores['by_site'][
                  'source_attribution']['full_vocab_accuracy'], 3),
              flush=True)

    model.eval()
    record('dense')
    for index, target in enumerate(targets):
        loss = phase(model, params, mask, train_data, args.device,
                     args.steps_per_round, args.batch_size, args.lr,
                     args.l1, args.seed+index)
        global_prune(params, mask, target)
        enforce(params, mask)
        record(f'pruned_{round(target*100)}', loss)
    loss = phase(model, params, mask, train_data, args.device,
                 args.final_steps, args.batch_size, args.lr, 0.,
                 args.seed+100)
    record('final_no_l1', loss)


if __name__ == '__main__':
    main()

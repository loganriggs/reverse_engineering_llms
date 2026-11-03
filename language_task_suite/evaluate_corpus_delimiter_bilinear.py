"""Re-evaluate saved bilinear delimiter models on the current held-out pairs."""

import argparse
import json
from pathlib import Path

import torch

from train_corpus_delimiter_bilinear import _model, evaluate, _forward


@torch.no_grad()
def evaluate_nuisance(model, pair, device, batch_size):
    if pair['base'] is None:
        return dict(n=0)
    model.eval()
    differences = []
    base_margins, changed_margins = [], []
    base, changed = pair['base'], pair['changed']
    for start in range(0, len(base['x']), batch_size):
        end = min(start+batch_size, len(base['x']))
        outputs = []
        for row in (base, changed):
            x = row['x'][start:end].to(device)
            site = row['site_position'][start:end].to(device)
            logits = _forward(model, x, device).float()
            ix = torch.arange(len(x), device=device)
            target = row['target_id'][start:end].to(device)
            alternative = row['alternative_target_id'][start:end].to(device)
            outputs.append(logits[ix, site, target] -
                           logits[ix, site, alternative])
        differences.extend((outputs[1]-outputs[0]).cpu().tolist())
        base_margins.extend(outputs[0].cpu().tolist())
        changed_margins.extend(outputs[1].cpu().tolist())
    values = torch.tensor(differences).abs()
    base_correct = torch.tensor(base_margins) > 0
    changed_correct = torch.tensor(changed_margins) > 0
    return dict(n=len(differences), mean_signed_margin_change=(
                    sum(differences)/len(differences)),
                mean_absolute_margin_change=values.mean().item(),
                median_absolute_margin_change=values.median().item(),
                p95_absolute_margin_change=torch.quantile(values, .95).item(),
                fraction_absolute_change_over_one=(values > 1).float().mean().item(),
                base_correct_fraction=base_correct.float().mean().item(),
                changed_correct_fraction=changed_correct.float().mean().item(),
                decision_flip_fraction=(base_correct != changed_correct).float().mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-path', type=Path, default=Path(
        '/home/loganriggs/Coding/neuron_circuits/artifacts/'
        'corpus_delimiter_pilot/dataset.pt'))
    p.add_argument('--checkpoint-dir', type=Path, default=Path(
        '/home/loganriggs/Coding/neuron_circuits/artifacts/'
        'corpus_delimiter_pilot/checkpoints'))
    p.add_argument('--depths', default='1,2,3,4')
    p.add_argument('--site-weight', type=float, default=50.)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    data = torch.load(args.cache_path, map_location='cpu', weights_only=True)
    device = torch.device(args.device)
    results = {}
    for depth in (int(x) for x in args.depths.split(',')):
        path = args.checkpoint_dir / f'depth{depth}_site{args.site_weight:g}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = _model(data['vocab_size'], data['n_ctx'], depth,
                       checkpoint['d_model'], checkpoint['heads'], device,
                       checkpoint.get('norm', 'rms'))
        model.load_state_dict(checkpoint['state_dict'])
        results[str(depth)] = dict(
            val=evaluate(model, data['val'], device, args.batch_size),
            test=evaluate(model, data['test'], device, args.batch_size),
            val_nuisance=evaluate_nuisance(
                model, data['val_nuisance'], device, args.batch_size),
            test_nuisance=evaluate_nuisance(
                model, data['test_nuisance'], device, args.batch_size),
            checkpoint_path=str(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        cache_path=str(args.cache_path), results=results), indent=2)+'\n')
    print({depth: (row['test']['both_two_way'],
                   row['test']['pure_both_top1'])
           for depth, row in results.items()}, flush=True)


if __name__ == '__main__':
    main()

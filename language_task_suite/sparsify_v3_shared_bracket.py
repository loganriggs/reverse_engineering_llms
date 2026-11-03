"""Iterative attention-weight pruning of a trained shared-bracket v3 model."""

import argparse
import json
from pathlib import Path

import torch

from sparsify_original_v3_for_rule_graphs import (enforce, global_prune,
    mask_stats, prunable, train_phase)
from eval_original_v3_shared_bracket import selected_pairs, score_pairs
from original_v3_shared_bracket import LanguageV3SharedBracket
from train_original_v3_shared_bracket import evaluate
from model import BilinearGPT, Config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--targets', default='.4,.6')
    p.add_argument('--steps-per-round', type=int, default=100)
    p.add_argument('--final-steps', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--l1', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=97801)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    payload = torch.load(args.source_run/'model.pt', map_location=device,
                         weights_only=True)
    model = BilinearGPT(Config(**payload['config'])).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    params = prunable(model)
    mask = {name: torch.ones_like(weight, dtype=torch.bool)
            for name, weight in params.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(stage):
        ordinary = evaluate(model, LanguageV3SharedBracket(seed=args.seed+1),
                            512, 129, 32, device, 128)
        pairs = selected_pairs(LanguageV3SharedBracket(seed=args.seed+2),
                               512, 129, 256)
        bracket = score_pairs(model, LanguageV3SharedBracket(seed=args.seed+3),
                              pairs, 32, device)
        row = dict(stage=stage, mask=mask_stats(mask),
                   per_rule_ce=ordinary['per_rule_ce'],
                   per_rule_n=ordinary['per_rule_n'],
                   full_bracket_oracle=bracket)
        rows.append(row)
        torch.save(dict(model=model.state_dict(), mask=mask,
                        config=payload['config']), args.output_dir/f'{stage}.pt')
        print(stage, 'sparsity', round(row['mask']['sparsity'],3),
              'noun', round(row['per_rule_ce']['noun_bigram'],3),
              'skip', round(row['per_rule_ce']['skip_bigram'],3),
              'bracket', round(row['per_rule_ce']['skip_trigram'],3),
              'oracle KL', round(bracket['all']['original_kl'],3), flush=True)

    record('dense')
    generator = LanguageV3SharedBracket(seed=args.seed)
    for target in map(float, args.targets.split(',')):
        train_phase(model, params, mask, generator, device,
                    args.steps_per_round, args.batch_size, 129,
                    args.lr, args.l1)
        global_prune(params, mask, target)
        enforce(params, mask)
        record(f'pruned_{round(target*100)}')
    train_phase(model, params, mask, generator, device, args.final_steps,
                args.batch_size, 129, args.lr, 0.)
    record('final_no_l1')
    (args.output_dir/'results.json').write_text(json.dumps(dict(
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        rows=rows,
        caveat='Only attention projection matrices are pruned; embeddings and '
               'unembedding remain dense. Fresh evaluations reuse the same DGP '
               'but independent random streams. This does not identify causal necessity.'
    ),indent=2)+'\n')


if __name__ == '__main__':
    main()

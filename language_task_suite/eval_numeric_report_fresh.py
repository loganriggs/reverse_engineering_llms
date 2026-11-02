"""Fresh exact-oracle KL panels for score-comparison checkpoints."""

import argparse
import json
import sys
from pathlib import Path

import torch

from numeric_report_language import GENRES, TASK_LABELS, NumericReportLanguage
from train_multi_genre import evaluate, make_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--depths', default='1,2,3,4')
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--length', type=int, default=63)
    p.add_argument('--seed', type=int, default=9901)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    panels = [make_eval(NumericReportLanguage(seed=a.seed+1009*i), a.n,
                        a.length, 'cpu', TASK_LABELS, GENRES)
              for i in range(a.repeats)]
    saved = json.loads((a.model_dir/'results.json').read_text())
    results = {}
    for depth in [int(x) for x in a.depths.split(',')]:
        record = next(r for r in saved['records'] if r['depth'] == depth)
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(a.model_dir/f'model_depth{depth}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        scores = [evaluate(model, panel) for panel in panels]
        aggregate = {}
        for label in scores[0]:
            count = sum(rep[label]['n'] for rep in scores)
            aggregate[label] = dict(n=count, **{
                field: sum(rep[label]['n']*rep[label][field] for rep in scores)
                       /count for field in ('kl', 'oracle_entropy', 'sampled_ce')})
        results[str(depth)] = dict(aggregate=aggregate, repeats=scores)
        print(depth, 'answer KL', aggregate['score_comparison_answer']['kl'],
              flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(
        config={**vars(a), 'model_dir': str(a.model_dir),
                'output': str(a.output)},
        results=results,
        key_blind_baseline_kl=0.6931471805599453,
        caveat='A Bayes control that sees both persons and score values but '
               'not which score belongs to which person predicts each person '
               'with probability one half. Fresh streams do not withhold '
               'person-score combinations from training.'), indent=2)+'\n')


if __name__ == '__main__':
    main()

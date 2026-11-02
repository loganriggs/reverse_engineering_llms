"""Evaluate saved bilinear multi-genre models on larger independent samples."""

import argparse
import json
import sys
from pathlib import Path

import torch

from multi_genre_language import MultiGenreLanguage
from train_multi_genre import evaluate, make_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--repeats', type=int, default=1,
                   help='Independent test streams of n sequences each')
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--seed', type=int, default=9901)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    results = {}
    for record in saved['records']:
        depth = record['depth']
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(Path(a.model_dir)/f'model_depth{depth}.pt',
                                         weights_only=True, map_location='cpu'))
        model.eval()
        replicate_scores = [evaluate(model, make_eval(
            MultiGenreLanguage(seed=a.seed+1009*i), a.n, a.length, 'cpu'))
            for i in range(a.repeats)]
        aggregate = {}
        for name in replicate_scores[0]:
            count = sum(rep[name]['n'] for rep in replicate_scores)
            aggregate[name] = dict(n=count)
            for field in ('kl', 'oracle_entropy', 'sampled_ce'):
                aggregate[name][field] = (
                    sum(rep[name]['n']*rep[name][field] for rep in replicate_scores
                        if rep[name][field] is not None)/count if count else None)
        results[str(depth)] = dict(aggregate=aggregate,
                                   repeats=replicate_scores)
        print('depth', depth,
              {name: round(aggregate[name]['kl'], 4)
               for name in ('report_mixed_verb', 'policy_mixed_action',
                            'faq_answer_copy', 'faq_place_recall',
                            'faq_new_place', 'list_number')}, flush=True)
    Path(a.output).write_text(json.dumps(dict(config=vars(a),
                                              results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()

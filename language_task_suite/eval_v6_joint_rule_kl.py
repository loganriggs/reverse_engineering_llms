"""Compare v6 checkpoints on a shared unrestricted exact-oracle stream panel."""

import argparse
import json
import sys
from pathlib import Path

import torch

from surface_mixture_v6 import SurfaceMixtureV6
from train_surface_mixture import evaluate, make_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--config-json', type=Path, required=True)
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    p.add_argument('--nstreams', type=int, default=1024)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--seed', type=int, default=23201)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads(a.config_json.read_text())
    config = (saved['model_config'] if 'model_config' in saved else
              next(item['model_config'] for item in saved['records']
                   if item['depth'] == 2))
    assert config['seq_len'] >= a.length
    gen = SurfaceMixtureV6(seed=a.seed, quote_rate=.30, source_rate=.30,
                           question_rate=.05, counterbalanced_queries=True)
    panel = make_eval(gen, a.nstreams, a.length, torch.device('cpu'))
    results = {}
    for checkpoint in a.checkpoints:
        model = BilinearGPT(Config(**config))
        model.load_state_dict(torch.load(checkpoint, map_location='cpu',
                                         weights_only=True))
        model.eval()
        results[str(checkpoint)] = evaluate(model, panel)
        print(checkpoint, {kind: round(results[str(checkpoint)][kind]['kl'], 4)
                           for kind in ('verb', 'object', 'question_answer',
                                        'source_answer')}, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(config={**vars(a),
                                              'config_json': str(a.config_json),
                                              'checkpoints': list(map(str, a.checkpoints)),
                                              'output': str(a.output)},
                                        results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()

"""Evaluate matched varied-language models on shared return-distance regimes."""

import argparse
import json
import sys
from pathlib import Path

import torch

from surface_mixture_v2_varied import SurfaceMixtureV2Varied
from train_surface_mixture import evaluate, make_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs', nargs='+', type=Path, required=True)
    p.add_argument('--n', type=int, default=512)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=20261209)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    regimes = dict(
        original={},
        return_train=dict(subject_repeat_mass=.1,
                          subject_return_mass=.3),
        return_stress=dict(subject_repeat_mass=.4,
                           subject_return_mass=.5))
    panels = {name: make_eval(SurfaceMixtureV2Varied(
        seed=a.seed, **kwargs), a.n, a.length, 'cpu')
              for name, kwargs in regimes.items()}
    output = dict(config={**vars(a),
                          'model_dirs': [str(x) for x in a.model_dirs],
                          'output': str(a.output)}, regimes=regimes,
                  results={})
    for directory in a.model_dirs:
        metadata = json.loads((directory/'results.json').read_text())
        assert metadata['config']['generator'] == 'v2_varied'
        record = next(row for row in metadata['records'] if row['depth'] == 2)
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(directory/'model_depth2.pt',
                                         weights_only=True, map_location='cpu'))
        model.eval()
        output['results'][str(directory)] = {}
        for name, panel in panels.items():
            scores = evaluate(model, panel)
            output['results'][str(directory)][name] = {
                key: float(scores[key]['kl']) for key in
                ('all', 'verb', 'verb_with_induction')}
            print(directory.name, name,
                  output['results'][str(directory)][name], flush=True)
    a.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()

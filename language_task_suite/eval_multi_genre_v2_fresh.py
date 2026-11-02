"""Independent exact-oracle KL panels for versioned Catalog models."""

import argparse
import json
import sys
from pathlib import Path

import torch

from multi_genre_v2_language import GENRES_V2, MultiGenreV2Language
from train_multi_genre import TASK_LABELS, evaluate, make_eval


TASK_LABELS_V2 = TASK_LABELS + (
    'catalog_record_object', 'catalog_owner_value', 'catalog_status_value',
    'catalog_query_object', 'catalog_owner_answer', 'catalog_status_answer',
    'catalog_first_owner_answer', 'catalog_first_status_answer')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--depths', default='', help='Default: every saved depth')
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--length', type=int, default=127)
    p.add_argument('--seed', type=int, default=9901)
    p.add_argument('--genre-weights', default='',
                   help='Optional evaluation weights; defaults to training weights')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    from softmax_attention_control import configure_attention

    saved = json.loads((a.model_dir/'results.json').read_text())
    assert saved['config']['language'] == 'v2'
    weights_text = a.genre_weights or saved['config'].get('genre_weights', '')
    weights = tuple(float(x) for x in weights_text.split(',')) if weights_text else None
    def generator(seed):
        kwargs = {'record_rate': saved['config'].get('catalog_record_rate', .60)}
        if weights is not None:
            kwargs['genre_weights'] = weights
        return MultiGenreV2Language(seed=seed, **kwargs)
    panels = [make_eval(generator(a.seed+1009*i), a.n, a.length, 'cpu',
                        TASK_LABELS_V2, GENRES_V2) for i in range(a.repeats)]
    requested = ({int(x) for x in a.depths.split(',')} if a.depths else None)
    results = {}
    for record in saved['records']:
        depth = record['depth']
        if requested is not None and depth not in requested:
            continue
        model = configure_attention(BilinearGPT(Config(**record['model_config'])),
                                    saved['config'].get('attention', 'bilinear'))
        model.load_state_dict(torch.load(a.model_dir/f'model_depth{depth}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        scores = [evaluate(model, panel) for panel in panels]
        aggregate = {}
        for name in scores[0]:
            count = sum(rep[name]['n'] for rep in scores)
            aggregate[name] = dict(n=count)
            for field in ('kl', 'oracle_entropy', 'sampled_ce'):
                aggregate[name][field] = (
                    sum(rep[name]['n']*rep[name][field] for rep in scores
                        if rep[name][field] is not None)/count if count else None)
        results[str(depth)] = dict(aggregate=aggregate, repeats=scores)
        print('depth', depth, {name: aggregate[name]['kl'] for name in
                              ('catalog_owner_answer', 'catalog_status_answer',
                               'catalog_first_owner_answer',
                               'catalog_first_status_answer')},
              flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(config={**vars(a),
                                            'model_dir': str(a.model_dir),
                                            'output': str(a.output)},
                                        results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()

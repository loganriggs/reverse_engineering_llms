"""Evaluate saved bilinear multi-genre models on larger independent samples."""

import argparse
import json
import sys
from pathlib import Path

import torch

from multi_genre_language import GENRES, MultiGenreLanguage
from train_multi_genre import TASK_LABELS, evaluate, make_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--repeats', type=int, default=1,
                   help='Independent test streams of n sequences each')
    p.add_argument('--length', type=int, default=None,
                   help='Defaults to the saved training context length')
    p.add_argument('--seed', type=int, default=9901)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    from softmax_attention_control import configure_attention

    torch.set_num_threads(4)
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    language = saved['config'].get('language', 'v1')
    if language == 'v1':
        generator_class, genres, task_labels = (MultiGenreLanguage,
                                                 GENRES, TASK_LABELS)
    elif language in ('v2', 'v3'):
        from multi_genre_v2_language import GENRES_V2, MultiGenreV2Language
        generator_class, genres = MultiGenreV2Language, GENRES_V2
        task_labels = TASK_LABELS + (
            'catalog_record_object', 'catalog_owner_value',
            'catalog_status_value', 'catalog_query_object',
            'catalog_owner_answer', 'catalog_status_answer',
            'catalog_first_owner_answer', 'catalog_first_status_answer')
        if language == 'v3':
            from multi_genre_v3_agreement import (GENRES_V3,
                                                 MultiGenreV3Agreement)
            generator_class, genres = MultiGenreV3Agreement, GENRES_V3
            task_labels += ('agreement_main_match',
                            'agreement_main_conflict', 'agreement_embedded')
    else:
        raise ValueError(f'Unsupported language {language}')
    weights = saved['config'].get('genre_weights', '')
    def generator(seed):
        kwargs = {'genre_weights': tuple(float(w) for w in weights.split(','))} \
            if weights else {}
        if language in ('v2', 'v3'):
            kwargs['record_rate'] = saved['config'].get('catalog_record_rate', .60)
        return generator_class(seed=seed, **kwargs)
    length = a.length if a.length is not None else saved['config']['length']
    results = {}
    for record in saved['records']:
        depth = record['depth']
        model = configure_attention(BilinearGPT(Config(**record['model_config'])),
                                    saved['config'].get('attention', 'bilinear'))
        model.load_state_dict(torch.load(Path(a.model_dir)/f'model_depth{depth}.pt',
                                         weights_only=True, map_location='cpu'))
        model.eval()
        replicate_scores = [evaluate(model, make_eval(
            generator(a.seed+1009*i), a.n, length, 'cpu', task_labels,
            genres))
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
                            'faq_new_place', 'list_number',
                            'agreement_main_conflict') if name in aggregate and
               aggregate[name]['kl'] is not None}, flush=True)
    Path(a.output).write_text(json.dumps(dict(config=vars(a),
                                              results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()

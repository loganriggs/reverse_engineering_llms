"""Test selective causal effects of removing first-layer token-tensor heads."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_multi_genre_edits import build_pairs
from eval_multi_genre_token_tensor import score_tables
from eval_multi_genre_token_tensor_truncation import evaluate
from multi_genre_language import MultiGenreLanguage
from train_multi_genre import make_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs', nargs='+', required=True)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--ntest', type=int, default=1024)
    p.add_argument('--nedit-streams', type=int, default=512)
    p.add_argument('--test-seed', type=int, default=9882)
    p.add_argument('--edit-seed', type=int, default=9883)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    gen = MultiGenreLanguage(seed=a.edit_seed)
    pairs = [pair for pair in build_pairs(gen, a.nedit_streams, a.length)
             if pair['kind'] in ('faq_answer_copy', 'faq_place_recall',
                                 'policy_condition', 'policy_nuisance_person')]
    assert any(pair['kind'] == 'faq_place_recall' for pair in pairs)
    test = make_eval(MultiGenreLanguage(seed=a.test_seed), a.ntest,
                     a.length, 'cpu')
    results = {}
    for directory in a.model_dirs:
        folder = Path(directory)
        saved = json.loads((folder/'results.json').read_text())
        record = next(row for row in saved['records'] if row['depth'] == a.depth)
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(folder/f'model_depth{a.depth}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(batch)[0].float()
                                for batch in test['x'].split(32)])
        reference = dict(logp=F.log_softmax(logits, -1))
        tables, downstream = score_tables(model, a.length)
        rows = []
        for removed in ((),) + tuple((h,) for h in range(tables.shape[0])) + (
                tuple(range(tables.shape[0])),):
            altered = tables.clone()
            for head in removed:
                altered[head].zero_()
            metrics = evaluate(model, altered, downstream, test,
                               reference, pairs, gen)
            rows.append(dict(removed_heads=list(removed), **metrics))
            print(folder.name, removed, 'FAQ edit L1',
                  metrics['edits']['faq_place_recall']['effect_l1_error'],
                  'Policy edit L1',
                  metrics['edits']['policy_condition']['effect_l1_error'], flush=True)
        results[folder.name] = dict(model_config=record['model_config'],
                                    rows=rows)
    a.output.write_text(json.dumps(dict(
        config={**vars(a), 'output': str(a.output)},
        edit_counts=dict(Counter(pair['kind'] for pair in pairs)),
        results=results,
        caveat='Ablates whole first-layer heads in the weight-only lag-table '
               'replay. The unablated row bounds lag approximation error. '
               'Taskwise model KL and oracle-edit error measure different '
               'effects; head selectivity is not unique circuit recovery.'),
        indent=2)+'\n')


if __name__ == '__main__':
    main()

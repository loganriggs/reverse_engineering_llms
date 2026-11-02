"""Fresh-panel rank sweep within heads selected by a prior ablation panel.

All unselected heads retain their full score tables. This measures whether a
low-rank approximation of one selected head preserves its causal task effect.
The selected head numbers were chosen on a disjoint development panel.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_multi_genre_edits import build_pairs
from eval_multi_genre_token_tensor import score_tables
from eval_multi_genre_token_tensor_truncation import evaluate, truncated_scores
from multi_genre_language import MultiGenreLanguage
from train_multi_genre import make_eval


# Selected before examining the fresh seed-9892/9893 panel. These are the
# strongest roughly FAQ-selective and broadly important layer-0 heads in the
# seed-9882/9883 whole-head ablation, respectively.
SELECTED = {
    'multi_genre_models_v1': (1, 0),
    'multi_genre_models_seed1': (3, 1),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs', nargs='+', required=True)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--ntest', type=int, default=1024)
    p.add_argument('--nedit-streams', type=int, default=512)
    p.add_argument('--test-seed', type=int, default=9892)
    p.add_argument('--edit-seed', type=int, default=9893)
    p.add_argument('--ranks', default='1,2,4,8,16,32')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    ranks = [int(x) for x in a.ranks.split(',')]
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
        assert folder.name in SELECTED
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
        baseline = evaluate(model, tables, downstream, test,
                            reference, pairs, gen)
        rows = []
        for head in SELECTED[folder.name]:
            for rank in ranks:
                altered = tables.clone()
                altered[head] = truncated_scores(tables[head:head+1], rank)[0]
                metrics = evaluate(model, altered, downstream, test,
                                   reference, pairs, gen)
                rows.append(dict(head=head, rank=rank, **metrics))
                print(folder.name, head, rank, 'FAQ edit L1',
                      metrics['edits']['faq_place_recall']['effect_l1_error'],
                      'Policy model KL',
                      metrics['groups']['policy_mixed_action']['model_to_truncated_kl'],
                      flush=True)
        results[folder.name] = dict(selected_heads=list(SELECTED[folder.name]),
                                    full_table_baseline=baseline, rows=rows)
    a.output.write_text(json.dumps(dict(
        config={**vars(a), 'output': str(a.output)},
        selection_panel=dict(test_seed=9882, edit_seed=9883,
                             source='multi_genre_layer0_token_tensor_head_ablation.json'),
        fresh_panel=dict(test_seed=a.test_seed, edit_seed=a.edit_seed,
                         edit_counts=dict(Counter(pair['kind'] for pair in pairs))),
        results=results,
        caveat='Rank is per relative-lag score matrix within one head. '
               'Other heads stay full. Head numbers were selected on a '
               'different panel; this is compression of a selected head, '
               'not evidence that individual singular vectors are stable '
               'concepts or an end-to-end circuit.'), indent=2)+'\n')


if __name__ == '__main__':
    main()

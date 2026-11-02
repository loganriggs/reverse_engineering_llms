"""Remove one score-matrix singular mode from an otherwise full layer-0 tensor.

Heads were selected on an earlier panel. Mode indices and lag bins are fixed
before evaluating the new stream/edit seeds. This tests necessity in the full
function, unlike rank-1 versus rank-2 truncation.
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
from eval_multi_genre_token_tensor_truncation import evaluate
from multi_genre_language import MultiGenreLanguage
from train_multi_genre import make_eval


SELECTED = {'multi_genre_models_v1': 1,
            'multi_genre_models_seed1': 3}
LAG_BINS = ((0, 7), (8, 15), (16, 31), (32, 70))


@torch.no_grad()
def singular_mode(table, index):
    u, s, vh = torch.linalg.svd(table.double(), full_matrices=False)
    component = (u[:, :, index, None]*s[:, index, None, None]*
                 vh[:, None, index, :])
    return component.float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs', nargs='+', required=True)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--ntest', type=int, default=1024)
    p.add_argument('--nedit-streams', type=int, default=512)
    p.add_argument('--test-seed', type=int, default=9912)
    p.add_argument('--edit-seed', type=int, default=9913)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    assert a.length == 71
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
        head = SELECTED[folder.name]
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
        for index in (0, 1, 2):
            component = singular_mode(tables[head], index)
            bins = ((0, a.length-1),) if index != 1 else (
                (0, a.length-1),) + LAG_BINS
            for start, end in bins:
                altered = tables.clone()
                altered[head, start:end+1] -= component[start:end+1]
                metrics = evaluate(model, altered, downstream, test,
                                   reference, pairs, gen)
                rows.append(dict(head=head, singular_index=index,
                                 lag_start=start, lag_end=end,
                                 score_energy_removed=float(
                                     component[start:end+1].square().sum()/
                                     tables[head].square().sum()),
                                 **metrics))
                print(folder.name, index, start, end,
                      'FAQ edit L1',
                      metrics['edits']['faq_place_recall']['effect_l1_error'],
                      'Policy model KL',
                      metrics['groups']['policy_mixed_action']['model_to_truncated_kl'],
                      flush=True)
        results[folder.name] = dict(head=head, full_table_baseline=baseline,
                                    rows=rows)
    a.output.write_text(json.dumps(dict(
        config={**vars(a), 'output': str(a.output)},
        selection_panel=dict(test_seed=9882, edit_seed=9883,
                             source='multi_genre_layer0_token_tensor_head_ablation.json'),
        hypothesis_panel=dict(test_seed=9892, edit_seed=9893,
                              source='multi_genre_layer0_token_tensor_selected_heads_fresh.json'),
        evaluation_panel=dict(test_seed=a.test_seed, edit_seed=a.edit_seed,
                              edit_counts=dict(Counter(pair['kind'] for pair in pairs))),
        results=results,
        caveat='A singular index denotes one rank-one term per relative '
               'lag within one head, not a single global vector. Modes are '
               'removed from the full score function; this tests selective '
               'necessity but does not establish a unique canonical circuit.'),
        indent=2)+'\n')


if __name__ == '__main__':
    main()

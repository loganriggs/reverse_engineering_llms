"""Rescore saved rank-one solutions after fixing the rare-seen mask.

The expensive optimization is not repeated. Only the incorrect metric field
is replaced; invariant task scores are checked against the saved values.
"""

import json

import numpy as np
import torch

from analyze_dual_output_language import factors, score
from analyze_toy_language import pair_table
from dual_joint_function_canonical import HERE
from train_dual_output_language import DualOutputAttention, exhaustive


def run():
    seq_t, targets_t, attrs = exhaustive()
    seq, targets = seq_t.numpy(), targets_t.numpy()
    for path in sorted(HERE.glob('dual_rank1_*.json')):
        saved = json.loads(path.read_text())
        model = DualOutputAttention().double()
        model.load_state_dict(torch.load(HERE / f"dual_output_seed{saved['seed']}.pt",
                                         map_location='cpu', weights_only=True))
        model.eval()
        heads = [factors(model, index) for index in range(2)]
        tables = [pair_table(head, np.asarray(matrix)) for head, matrix in
                  zip(heads, saved['best']['matrices'])]
        corrected = score(tables, seq, targets,
                          model.read.bias.detach().numpy(), attrs)[0]
        previous = saved['best']['score']
        for field in ('joint_accuracy', 'bit_accuracy'):
            assert previous[field] == corrected[field], (path, field)
        for field in ('common', 'rare_heldout'):
            for metric in ('n', 'joint_accuracy', 'bit_accuracy'):
                assert previous[field][metric] == corrected[field][metric], \
                    (path, field, metric)
        # Saved rank-one matrices were rounded, so BCE can differ slightly.
        assert abs(previous['bce'] - corrected['bce']) < 1e-3, path
        previous['rare_seen'] = corrected['rare_seen']
        path.write_text(json.dumps(saved, indent=2) + '\n')
        print('rescored', path.name)


if __name__ == '__main__':
    run()

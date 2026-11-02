"""Independent rule-edit test of sparse architecture distillation checkpoints."""

import argparse
import json
import sys
from pathlib import Path

import torch

from distill_sparse_depth_vs_factor import build_students
from eval_surface_mixture_edits import predict, score
from eval_surface_mixture_v2_edits import build_pairs
from surface_mixture_v2_varied import SurfaceMixtureV2Varied


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--teacher-dir', type=Path, default=Path('language_task_suite/surface_mixture_v2_varied_soft_models'))
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--arms', default='flat,factor,depth_split')
    p.add_argument('--edge-budgets', default='29491,14746')
    p.add_argument('--nstreams', type=int, default=256)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=95631)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    record_file = json.loads((a.teacher_dir / 'results.json').read_text())
    record = next(r for r in record_file['records'] if r['depth'] == 2)
    teacher = BilinearGPT(Config(**record['model_config']))
    teacher.load_state_dict(torch.load(a.teacher_dir / 'model_depth2.pt',
                                       map_location='cpu', weights_only=True))
    teacher.eval()
    gen = SurfaceMixtureV2Varied(seed=a.seed,
                                 **record_file['generator_kwargs'])
    pairs = build_pairs(gen, a.nstreams, a.length)
    original = [pair['original'] for pair in pairs]
    edited = [pair['edited'] for pair in pairs]
    models = {'teacher': teacher, **build_students(teacher, BilinearGPT, .01)}
    results = {}
    # Checkpoint names encode density within the arm, whose raw edge count
    # differs between the two-layer flat and the expanded students.
    for label in ['teacher'] + [f'{arm}:{budget}'
                                for arm in a.arms.split(',')
                                for budget in a.edge_budgets.split(',')]:
        if label == 'teacher':
            model = teacher
        else:
            arm, budget = label.split(':')
            model = models[arm]
            raw = sum(p.numel() for name, p in model.named_parameters()
                      if '.attn.' in name)
            path = a.run_dir / f'{arm}_{int(budget)/raw:.4f}.pt'
            saved = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model'])
            model.eval()
        before = predict(model, original, len(gen.vocab))
        after = predict(model, edited, len(gen.vocab))
        results[label] = score(pairs, before, after, gen)
        print(label, {kind: round(row['effect_l1_error'], 3)
                      for kind, row in results[label].items()}, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(config={**vars(a),
                                             'teacher_dir': str(a.teacher_dir),
                                             'run_dir': str(a.run_dir),
                                             'output': str(a.output)},
                                       pair_count=len(pairs), results=results),
                                   indent=2)+'\n')


if __name__ == '__main__':
    main()

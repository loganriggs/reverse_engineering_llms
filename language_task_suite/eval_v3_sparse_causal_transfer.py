"""Independent prediction-fidelity panels for causal-transfer students."""

import argparse
import json
import sys
from pathlib import Path

import torch

from distill_sparse_depth_vs_factor import (build_students, evaluation_panels,
                                            generator_for_teacher, panel_scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--teacher-dir', type=Path,
                   default=Path('language_task_suite/surface_mixture_v3_object8'))
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--streams', type=int, default=256)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    teacher_record = json.loads((a.teacher_dir/'results.json').read_text())
    record = next(r for r in teacher_record['records'] if r['depth'] == 2)
    teacher = BilinearGPT(Config(**record['model_config']))
    teacher.load_state_dict(torch.load(a.teacher_dir/'model_depth2.pt',
                                       weights_only=True, map_location='cpu'))
    teacher.eval().requires_grad_(False)
    generator, kwargs = generator_for_teacher(teacher_record)
    panels = evaluation_panels(generator, kwargs, a.streams, a.length)
    result = dict(config=dict(teacher_dir=str(a.teacher_dir),
                              run_dir=str(a.run_dir), streams=a.streams,
                              length=a.length),
                  panels={})
    for label in ('flat_kl', 'flat_causal', 'factor_kl', 'factor_causal'):
        model = build_students(teacher, BilinearGPT, .01)[label.split('_')[0]]
        saved = torch.load(a.run_dir/f'{label}.pt', weights_only=False,
                           map_location='cpu')
        model.load_state_dict(saved['model'])
        model.eval().requires_grad_(False)
        result['panels'][label] = {name: panel_scores(model, teacher, panel)
                                   for name, panel in panels.items()}
        print(label, {name: value['teacher_kl'] for name, value in
                      result['panels'][label].items()}, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

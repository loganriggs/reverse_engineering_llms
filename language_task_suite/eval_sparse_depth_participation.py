"""Test whether each block in a sparse depth student functionally participates."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from distill_sparse_depth_vs_factor import (build_students, evaluation_panels,
                                            generator_for_teacher)


@torch.no_grad()
def mean_kl(left, right):
    logp = F.log_softmax(left.float(), -1)
    return float((logp.exp()*(logp-F.log_softmax(right.float(), -1)))
                 .sum(-1).mean())


@torch.no_grad()
def evaluate(model, teacher, x):
    native, ablated = [], [[] for _ in model.blocks]
    target = []
    for batch in x.split(16):
        target.append(teacher(batch)[0])
        native_logits = model(batch)[0]
        native.append(native_logits)
        for block_index, block in enumerate(model.blocks):
            handle = block.attn.out.register_forward_hook(
                lambda _module, _inputs, output: torch.zeros_like(output))
            try:
                ablated[block_index].append(model(batch)[0])
            finally:
                handle.remove()
    target = torch.cat(target)
    native = torch.cat(native)
    native_teacher_kl = mean_kl(target, native)
    rows = []
    for logits in ablated:
        logits = torch.cat(logits)
        rows.append(dict(native_to_ablated_kl=mean_kl(native, logits),
                         teacher_to_ablated_kl=mean_kl(target, logits),
                         teacher_kl_increase=mean_kl(target, logits)-native_teacher_kl))
    return dict(native_teacher_kl=native_teacher_kl, blocks=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--teacher-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--arm', default='depth_perturbed')
    p.add_argument('--streams', type=int, default=256)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    record_file = json.loads((a.teacher_dir/'results.json').read_text())
    record = next(row for row in record_file['records'] if row['depth'] == 2)
    teacher = BilinearGPT(Config(**record['model_config']))
    teacher.load_state_dict(torch.load(a.teacher_dir/'model_depth2.pt',
                                       map_location='cpu', weights_only=True))
    teacher.eval().requires_grad_(False)
    model = build_students(teacher, BilinearGPT, .01)[a.arm]
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    model.eval().requires_grad_(False)
    generator, kwargs = generator_for_teacher(record_file)
    panels = evaluation_panels(generator, kwargs, a.streams, a.length)
    result = dict(config=dict(teacher_dir=str(a.teacher_dir),
                              checkpoint=str(a.checkpoint), arm=a.arm,
                              streams=a.streams, length=a.length),
                  active_edges={str(i): sum(int(mask.sum()) for name, mask in
                      saved['masks'].items() if name.startswith(f'blocks.{i}.attn.'))
                      for i in range(len(model.blocks))},
                  panels={name: evaluate(model, teacher, panel['x'])
                          for name, panel in panels.items()})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['panels'], indent=2), flush=True)


if __name__ == '__main__':
    main()

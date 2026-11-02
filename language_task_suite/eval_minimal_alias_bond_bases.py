"""Compare bond bases on a competent keyed-alias bilinear model.

The basis fits use only train-allowed alias-person pairs. Every reported test
uses pairs absent from training. This is local compression/mediation, not a
claim to have recovered a unique circuit.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from minimal_alias_lookup import NPERSON, all_cases, batch, evaluate, predict


@torch.no_grad()
def capture(model, bond, cases, edit, device, batch_size=512):
    out = []
    def hook(_, inputs):
        out.append(inputs[0][:, -1].detach().float().cpu())
    handle = bond.register_forward_pre_hook(hook)
    for start in range(0, len(cases), batch_size):
        x, _ = batch(cases[start:start+batch_size], edit, device)
        model(x)
    handle.remove()
    return torch.cat(out)


def bases(model, bond, device):
    fit = all_cases('train')
    raw = capture(model, bond, fit, 'none', device)
    changed = capture(model, bond, fit, 'query', device)
    dim = raw.shape[-1]
    centered = torch.cat([raw, changed])
    centered -= centered.mean(0, keepdim=True)
    _, _, data_vh = torch.linalg.svd(centered.double(), full_matrices=False)
    _, _, response_vh = torch.linalg.svd((changed-raw).double(),
                                         full_matrices=False)
    _, _, weight_vh = torch.linalg.svd(bond.weight.detach().float().cpu().double(),
                                      full_matrices=False)
    output_weight = model.lm_head.weight.detach().float().cpu().double()
    bond_weight = bond.weight.detach().float().cpu().double()
    _, _, all_output_vh = torch.linalg.svd(output_weight@bond_weight,
                                           full_matrices=True)
    _, _, person_output_vh = torch.linalg.svd(output_weight[:NPERSON]@bond_weight,
                                              full_matrices=True)
    random, _ = np.linalg.qr(np.random.default_rng(6511).normal(size=(dim, dim)))
    return dict(weight_svd=weight_vh.T.float(),
                all_output=all_output_vh.T.float(),
                person_output=person_output_vh.T.float(),
                activation_pca=data_vh.T.float(),
                query_response=response_vh.T.float(),
                random=torch.tensor(random, dtype=torch.float32),
                physical=torch.eye(dim))


def score(model, cases, reference, device):
    pred, _ = predict(model, cases, 'none', device)
    behavioral = evaluate(model, cases, device)
    behavioral['reference_tv'] = float((pred-reference).abs().sum(-1).mean()/2)
    return behavioral


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path,
                   default=Path('language_task_suite/minimal_alias_bilinear_v2'))
    p.add_argument('--layer', type=int, default=1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads((a.model_dir/'results.json').read_text())
    if saved['config']['attention'] != 'bilinear':
        raise ValueError('This experiment expects a bilinear checkpoint')
    device = torch.device(a.device)
    model = BilinearGPT(Config(**saved['model_config'])).to(device)
    model.load_state_dict(torch.load(a.model_dir/'model.pt', map_location=device,
                                     weights_only=True))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    basis = bases(model, bond, device)
    panels = {name: all_cases(name) for name in ('id_test', 'pair_holdout')}
    with torch.no_grad():
        reference = {name: predict(model, cases, 'none', device)[0]
                     for name, cases in panels.items()}
        baseline = {name: score(model, cases, reference[name], device)
                    for name, cases in panels.items()}
    widths = (0, 2, 4, 8, 16, 32, 48, 64)
    rows = []
    for name, directions in basis.items():
        assert torch.linalg.matrix_norm(directions.T@directions-torch.eye(64)) < 1e-4
        for width in widths:
            uk = directions[:, :width].to(device)
            def truncate(_, inputs):
                x = inputs[0]
                out = x.clone()
                out[:, -1] = (x[:, -1]@uk)@uk.T
                return (out,)
            handle = bond.register_forward_pre_hook(truncate)
            with torch.no_grad():
                scores = {panel: score(model, cases, reference[panel], device)
                          for panel, cases in panels.items()}
            handle.remove()
            if width == 64:
                assert all(row['reference_tv'] < 1e-5 for row in scores.values())
            rows.append(dict(basis=name, width=width, scores=scores))
            if width in (0, 4, 8, 16, 32, 64):
                row = scores['pair_holdout']
                print(name, width, round(row['accuracy'], 4),
                      round(row['edits']['query']['effect_l1_error'], 4),
                      round(row['reference_tv'], 5), flush=True)
    result = dict(config={**vars(a), 'model_dir': str(a.model_dir),
                          'output': str(a.output)},
                  fit_cases=len(all_cases('train')),
                  panels={name: len(cases) for name, cases in panels.items()},
                  widths=widths, baseline=baseline, rows=rows,
                  interpretation='Full width is an exact orthogonal gauge. '
                                 'Truncation tests prediction and intervention '
                                 'fidelity at the final attention output bond. '
                                 'Query-response directions are supervised by '
                                 'training-pair edits and are not task-blind.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

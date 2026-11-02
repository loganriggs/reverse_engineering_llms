"""Locate joint-language behavior across trained attention blocks."""

import argparse
import json
import sys
from pathlib import Path

import torch

from eval_joint_shared_token_basis import edit_scores, prediction_scores
from eval_surface_mixture_v2_nested_edits import build_pairs as bracket_pairs
from surface_mixture_v3_nested_varlen import SurfaceMixtureV3NestedVarlen
from train_surface_mixture import make_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--model-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--nstreams', type=int, default=128)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261290)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    sys.path.insert(0, args.source)
    from model import BilinearGPT, Config
    result = dict(config={**vars(args),
                          'model_dirs':[str(p) for p in args.model_dirs],
                          'output':str(args.output)}, models={},
                  caveat='Zeroing an attention output is a finite block '
                         'intervention. Other blocks remain active; a small '
                         'single-block effect can reflect redundancy, and '
                         'a large effect does not identify an internal '
                         'algorithm. Same grammar-valid bracket edits are '
                         'used across both model seeds.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for folder in args.model_dirs:
        metadata = json.loads((folder/'results.json').read_text())
        kwargs = metadata['generator_kwargs']
        record = next(row for row in metadata['records']
                      if row['depth'] == args.depth)
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(folder/f'model_depth{args.depth}.pt',
                                         map_location='cpu',weights_only=True))
        model = model.to(device).eval()
        test = make_eval(SurfaceMixtureV3NestedVarlen(
            seed=args.seed, **kwargs),args.nstreams,args.length,device)
        gen = SurfaceMixtureV3NestedVarlen(seed=args.seed+1,**kwargs)
        pairs = bracket_pairs(gen,args.nstreams,args.length)
        native_prediction,_ = prediction_scores(model,test)
        native_edits = edit_scores(model,gen,pairs,device=device)
        rows = {}
        for index, block in enumerate(model.blocks):
            handle = block.attn.out.register_forward_hook(
                lambda _module,_inputs,output: torch.zeros_like(output))
            try:
                prediction,_ = prediction_scores(model,test)
                edits = edit_scores(model,gen,pairs,device=device)
            finally:
                handle.remove()
            rows[f'block_{index}'] = dict(prediction=prediction,edits=edits)
        result['models'][folder.name] = dict(
            native=dict(prediction=native_prediction,edits=native_edits),
            ablated=rows)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        for name,row in rows.items():
            print(folder.name,name,'outer KL',round(
                row['prediction']['outer']['oracle_kl'],4),
                'outer edit L1',round(
                    row['edits']['outer_relevant']['effect_l1'],4),flush=True)


if __name__ == '__main__':
    main()

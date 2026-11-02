"""Evaluate all attention-block subsets on joint-language rule effects."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

from eval_joint_basis_mode_ablations import pairs_for, selected_edit_scores
from eval_joint_shared_token_basis import prediction_scores
from surface_mixture_v3_nested_varlen import SurfaceMixtureV3NestedVarlen
from train_surface_mixture import make_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--model-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--nstreams', type=int, default=128)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261300)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    device = torch.device(args.device)
    result = dict(config={**vars(args),
                          'model_dirs':[str(p) for p in args.model_dirs],
                          'output':str(args.output)}, models={},
                  caveat='A kept block receives the altered residual stream '
                         'produced by any earlier zeroed blocks. Subset '
                         'scores show finite sufficiency in this ablation '
                         'family, not a unique minimal algorithm or an '
                         'in-distribution retrained subnetwork. Prediction '
                         'KL, relevant effects, and nuisance effects must '
                         'be read separately.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for folder in args.model_dirs:
        metadata = json.loads((folder/'results.json').read_text())
        assert metadata['config']['generator'] == 'v3_nested_varlen'
        kwargs = metadata['generator_kwargs']
        record = next(row for row in metadata['records']
                      if row['depth'] == args.depth)
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(folder/f'model_depth{args.depth}.pt',
                                         map_location='cpu',weights_only=True))
        model = model.to(device).eval()
        test = make_eval(SurfaceMixtureV3NestedVarlen(
            seed=args.seed, **kwargs),args.nstreams,args.length,device)
        pairs = pairs_for(args.seed+1,args.nstreams,args.length,kwargs)
        native_prediction,native_logp = prediction_scores(model,test)
        native_edits = selected_edit_scores(model,pairs,None,None,device)
        rows = {}
        for bits in itertools.product((0,1),repeat=args.depth):
            name = ''.join(str(bit) for bit in bits)
            handles = [model.blocks[i].attn.out.register_forward_hook(
                lambda _module,_inputs,output: torch.zeros_like(output))
                for i,bit in enumerate(bits) if bit == 0]
            try:
                prediction,_ = prediction_scores(model,test,
                                                  native_logp=native_logp)
                edits = selected_edit_scores(model,pairs,None,None,device)
            finally:
                for handle in handles:
                    handle.remove()
            rows[name] = dict(kept_blocks=[i for i,bit in enumerate(bits)
                                          if bit],prediction=prediction,
                              edits=edits)
            print(folder.name,name,'outer KL',round(
                prediction['outer']['oracle_kl'],4),
                'outer edit L1',round(
                    edits['bracket']['outer_relevant']['effect_l1'],4),
                flush=True)
        full = '1'*args.depth
        assert rows[full]['prediction']['all']['teacher_kl'] < 1e-8
        result['models'][folder.name] = dict(
            native=dict(prediction=native_prediction,edits=native_edits),
            subsets=rows)
        args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()

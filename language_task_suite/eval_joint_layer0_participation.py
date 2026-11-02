"""Does the first attention layer causally carry each joint-language task?"""

import argparse
import json
import sys
from pathlib import Path

import torch

from eval_joint_basis_mode_ablations import pairs_for, selected_edit_scores
from eval_joint_shared_token_basis import prediction_scores
from eval_multi_genre_token_tensor import score_tables
from surface_mixture_v3_nested_varlen import SurfaceMixtureV3NestedVarlen
from train_surface_mixture import make_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--model-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--nstreams', type=int, default=64)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261280)
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
                  caveat='Only the exact first-layer token-score table is '
                         'zeroed, by head or entirely. Later attention '
                         'layers stay native; this tests participation, '
                         'not an isolated mechanism or a minimal circuit.')
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
        model.blocks[0].attn.rotary.inv_freq = (
            model.blocks[0].attn.rotary.inv_freq.to(device))
        test = make_eval(SurfaceMixtureV3NestedVarlen(
            seed=args.seed, **kwargs),args.nstreams,args.length,device)
        pairs = pairs_for(args.seed+1,args.nstreams,args.length,kwargs)
        tables,downstream = score_tables(model,args.length)
        native,logp = prediction_scores(model,test)
        rows = {}
        for name,removed_heads in [('full_table',()),
            *[(f'head_{head}',(head,)) for head in range(tables.shape[0])],
            ('all_heads',tuple(range(tables.shape[0])))]:
            changed = tables.clone()
            for head in removed_heads:
                changed[head] = 0.
            prediction,_ = prediction_scores(model,test,changed,
                                              downstream,logp)
            edits = selected_edit_scores(model,pairs,changed,
                                         downstream,device)
            rows[name] = dict(prediction=prediction,edits=edits)
        assert rows['full_table']['prediction']['all']['teacher_kl'] < 1e-4
        result['models'][folder.name] = dict(native=native,rows=rows)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        for name in ('head_0','head_1','head_2','head_3','all_heads'):
            row=rows[name]
            print(folder.name,name,'KL verb/object/outer',
                  *[round(row['prediction'][k]['oracle_kl'],4)
                    for k in ('verb','object','outer')],
                  'outer edit L1',round(
                    row['edits']['bracket']['outer_relevant']['effect_l1'],4),
                  flush=True)


if __name__ == '__main__':
    main()

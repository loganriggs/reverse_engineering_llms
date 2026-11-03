"""Compare causal source-edit response after one attention-head output ablation."""

import argparse
import json
from pathlib import Path

import torch

from eval_document_world_v5_clause_causal import collect,score,DeepModel,SPECS


def load_model(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    args=saved['model_kwargs']
    model=DeepModel(len(saved['vocabulary']),args['d_model'],args['n_head'],
                    SPECS[f"block{saved['depth']}"],saved['n_ctx'],
                    norm=args['norm'],residual=args['residual'],
                    attention=args['attention']).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model,saved


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--streams',type=int,default=1000)
    p.add_argument('--max-cases',type=int,default=128)
    p.add_argument('--seed',type=int,default=341991)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    models=[load_model(path,a.device) for path in a.checkpoints]
    ref=models[0][1]
    assert all(saved['vocabulary']==ref['vocabulary'] and
               saved['generator_kwargs']==ref['generator_kwargs'] and
               saved['n_ctx']==ref['n_ctx'] for _,saved in models)
    all_cases,skipped=collect(a.streams,a.seed,ref['generator_kwargs'],
                              ref['n_ctx'],a.max_cases)
    cases={key:all_cases[key] for key in
           ('source_attribution','source_nuisance')}
    rows={}
    for path,(model,saved) in zip(a.checkpoints,models,strict=True):
        base=score(model,cases,ref['vocabulary'],ref['n_ctx'],a.device,
                   a.batch_size)
        heads=[]
        for layer_index,layer in enumerate(model.layers):
            if not hasattr(layer,'o'):
                continue
            width=layer.d_head
            for head_index in range(layer.n_head):
                sl=slice(head_index*width,(head_index+1)*width)
                with torch.no_grad():
                    original=layer.o.weight[:,sl].clone()
                    layer.o.weight[:,sl]=0.
                try:
                    ablated=score(model,cases,ref['vocabulary'],ref['n_ctx'],
                                  a.device,a.batch_size)
                finally:
                    with torch.no_grad():
                        layer.o.weight[:,sl]=original
                heads.append(dict(layer=layer_index,head=head_index,
                                  source_response=ablated['source_attribution']
                                  ['mean_model_response'],
                                  source_response_drop=base['source_attribution']
                                  ['mean_model_response']-ablated
                                  ['source_attribution']['mean_model_response'],
                                  nuisance_response=ablated['source_nuisance']
                                  ['mean_model_response']))
        rows[str(path)]=dict(baseline=base,head_ablations=heads)
        print(path,'baseline',base['source_attribution']
              ['mean_model_response'],'top drops',sorted(
                  ((r['layer'],r['head'],r['source_response_drop'])
                   for r in heads),key=lambda row:-row[2])[:4],flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(config={**vars(a),
        'checkpoints':[str(v) for v in a.checkpoints],
        'output':str(a.output)},skipped=skipped,models=rows,
        caveat='Zeroing a head output tests necessity under one intervention; '
               'it does not identify the QK/OV token path or prove a sparse '
               'model is more interpretable.'),indent=2)+'\n')


if __name__=='__main__':main()

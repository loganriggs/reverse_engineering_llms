"""Score v8 rule families on the same fresh exact-oracle streams."""

import argparse
import json
import sys
from pathlib import Path

import torch

from surface_mixture_v8 import SurfaceMixtureV8
from train_surface_mixture import evaluate, make_eval


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs',nargs='+',required=True)
    p.add_argument('--n',type=int,default=256)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--seed',type=int,default=38101)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config
    folders=[Path(name) for name in a.model_dirs]
    saved=[json.loads((folder/'results.json').read_text()) for folder in folders]
    kwargs=saved[0]['generator_kwargs']
    assert all(item['generator_kwargs']==kwargs for item in saved)
    gen=SurfaceMixtureV8(seed=a.seed,**kwargs)
    data=make_eval(gen,a.n,a.length,torch.device(a.device))
    out={}
    for folder,record in zip(folders,saved):
        model_record=record['records'][0]
        model=BilinearGPT(Config(**model_record['model_config'])).to(a.device)
        model.load_state_dict(torch.load(folder/f"model_depth{model_record['depth']}.pt",
                                         map_location=a.device,weights_only=True))
        model.eval()
        scores=evaluate(model,data)
        out[folder.name]=scores
        print(folder.name,{key:round(scores[key]['kl'],4)
                           for key in ('verb','object','question_answer',
                                       'source_answer','scope_answer')
                           if key in scores},flush=True)
    result=dict(config=vars(a),generator_kwargs=kwargs,results=out,
                caveat='All models share the same fresh v8 exact-oracle panel. '
                       'This is prediction KL, not causal edit fidelity; '
                       'scope key-use is evaluated separately.')
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()

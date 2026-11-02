"""Score a short scope pretrainer directly on the long v8 task distribution."""

import argparse
import json
import sys
from pathlib import Path

import torch

from eval_surface_mixture_edits import predict
from eval_surface_mixture_v7_scope import build,prediction
from surface_mixture_v8 import SurfaceMixtureV8


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir',required=True)
    p.add_argument('--seed',type=int,default=9101)
    p.add_argument('--n',type=int,default=600)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--max-questions',type=int,default=200)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config
    folder=Path(a.model_dir)
    saved=json.loads((folder/'results.json').read_text())
    gen=SurfaceMixtureV8(seed=a.seed,**saved['generator_kwargs'])
    questions,_,_=build(gen,a.n,a.length,a.max_questions)
    model=BilinearGPT(Config(**saved['model_config']))
    model.load_state_dict(torch.load(folder/'model_depth3.pt',
                                     map_location='cpu',weights_only=True))
    model.eval()
    probs=predict(model,[q['prefix'] for q in questions],len(gen.vocab))
    result=dict(config=vars(a),selected_pretrain_step=saved['selected_step'],
                prediction=prediction(questions,probs),
                caveat='Long v8 task-rich test without joint fine-tuning; '
                       'fresh stream seed but no section-color pair holdout.')
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(result['prediction'],flush=True)


if __name__=='__main__':
    main()

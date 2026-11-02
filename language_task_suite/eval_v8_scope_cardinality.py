"""Audit section-key behavior by number of completed sections."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eval_surface_mixture_edits import predict
from eval_surface_mixture_v7_scope import build
from surface_mixture_v8 import SurfaceMixtureV8


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs',nargs='+',required=True)
    p.add_argument('--seed',type=int,default=38201)
    p.add_argument('--n',type=int,default=600)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--max-questions',type=int,default=300)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config
    folders=[Path(x) for x in a.model_dirs]
    saved=[json.loads((f/'results.json').read_text()) for f in folders]
    kwargs=saved[0]['generator_kwargs']
    assert all(x['generator_kwargs']==kwargs for x in saved)
    gen=SurfaceMixtureV8(seed=a.seed,**kwargs)
    questions,pairs,_=build(gen,a.n,a.length,a.max_questions)
    key_pairs=[pair for pair in pairs if pair['kind']=='section_key']
    key_by_prefix={tuple(pair['original']):pair for pair in key_pairs}
    assert all(tuple(q['prefix']) in key_by_prefix for q in questions)
    groups=defaultdict(list)
    for i,q in enumerate(questions):
        groups[round(1/q['key_blind_bayes'])].append(i)
    output={}
    for folder,record in zip(folders,saved):
        model_record=record['records'][0]
        model=BilinearGPT(Config(**model_record['model_config']))
        model.load_state_dict(torch.load(folder/f"model_depth{model_record['depth']}.pt",
                                         map_location='cpu',weights_only=True))
        model.eval()
        p0=predict(model,[q['prefix'] for q in questions],len(gen.vocab))
        p1=predict(model,[key_by_prefix[tuple(q['prefix'])]['edited']
                          for q in questions],len(gen.vocab))
        result={}
        for count,indices in sorted(groups.items()):
            ids=np.asarray(indices)
            targets=np.asarray([questions[i]['target'] for i in ids])
            effect=p1[ids]-p0[ids]
            result[str(count)]=dict(n=len(ids),
                answer_accuracy=float(np.mean(p0[ids].argmax(-1)==targets)),
                answer_ce=float(np.mean(-np.log(np.maximum(
                    p0[ids,targets],1e-30)))),
                key_blind_accuracy=1/count,key_blind_ce=float(np.log(count)),
                key_edit_tv=float(np.abs(effect).sum(-1).mean()/2))
        output[folder.name]=result
        print(folder.name,result,flush=True)
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(config=vars(a),generator_kwargs=kwargs,
                                    results=output),indent=2)+'\n')


if __name__=='__main__':
    main()

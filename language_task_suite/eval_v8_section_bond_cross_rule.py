"""Test section-fitted attention-bond featurizers on distinct v8 source rules."""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

from eval_v8_rule_holdout_readout import source_panel
from surface_mixture_v8 import SurfaceMixtureV8
from train_v8_section_combo_featurizers import add_targets, evaluate, prepare_early
from train_surface_v2_shared_groups import ScalarTopK


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--feature-source',default='/home/loganriggs/Coding/neuron_circuits/scripts')
    p.add_argument('--model-dir',required=True)
    p.add_argument('--site',choices=('layer0_bond','layer1_bond'),required=True)
    p.add_argument('--checkpoint',action='append',required=True)
    p.add_argument('--seed',type=int,default=85001)
    p.add_argument('--streams',type=int,default=600)
    p.add_argument('--cap',type=int,default=80)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0,a.source)
    sys.path.insert(0,a.feature_source)
    from feature_structures.models import SharedGroups
    from model import BilinearGPT,Config
    folder=Path(a.model_dir)
    saved=json.loads((folder/'results.json').read_text())
    record=saved['records'][0]
    device=torch.device(a.device)
    model=BilinearGPT(Config(**record['model_config'])).to(device)
    model.load_state_dict(torch.load(folder/f"model_depth{record['depth']}.pt",
                                     map_location=device,weights_only=True))
    model.eval().requires_grad_(False)
    generator=SurfaceMixtureV8(seed=a.seed,**saved['generator_kwargs'])
    pairs,attempted=source_panel(a.seed,saved['generator_kwargs'],
                                 a.streams,a.cap)
    assert pairs and {pair['kind'] for pair in pairs}=={
        'other_speaker','query_object','statement_speaker'}
    assert all(len(pair['q_before'])==len(pair['q_after'])==1 for pair in pairs)
    data=add_targets(prepare_early(model,pairs,generator.token2id,
                    int(a.site[5]),generator.token2id['.'],None,None),
                     pairs,generator.token2id)
    checkpoints=[torch.load(path,map_location=device,weights_only=True)
                 for path in a.checkpoint]
    center,scale=checkpoints[0]['center'],checkpoints[0]['scale']
    assert all(torch.allclose(center,cp['center'],atol=1e-5,rtol=1e-5)
               and torch.allclose(scale,cp['scale'],atol=1e-5,rtol=1e-5)
               for cp in checkpoints)
    rows={'native':evaluate(model,data,center,scale,site=a.site)}
    weight=model.blocks[int(a.site[5])].attn.out.weight
    _,_,vh=torch.linalg.svd(weight.double(),full_matrices=False)
    for rank in (8,10,16):
        rows[f'weight_svd{rank}']=evaluate(model,data,center,scale,
                                           basis=vh.T.float(),k=rank,
                                           site=a.site)
    for cp in checkpoints:
        for name,state in cp['states'].items():
            if name in rows:
                raise ValueError(f'Duplicate featurizer name {name}')
            if name.startswith('scalar'):
                rank=int(re.match(r'scalar(\d+)',name).group(1))
                feature=ScalarTopK(64,64,rank).to(device)
                membership=None
            else:
                feature=SharedGroups(64,64,16,4,4).to(device)
                membership=(cp['fixed_membership']
                            if name.startswith('disjoint') else None)
            feature.load_state_dict(state)
            feature.eval()
            rows[name]=evaluate(model,data,center,scale,feature=feature,
                                membership=membership,site=a.site)
    result=dict(config=vars(a),counts=dict(Counter(pair['kind'] for pair in pairs)),
                attempted=dict(attempted),
                max_endpoint_replay_error=data['replay_error'],rows=rows,
                caveat='The source task family was withheld from featurizer '
                       'fit and development, but was included in frozen LM '
                       'training. This tests transfer of a section-fitted '
                       'local representation, not new-task discovery.')
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('source counts',result['counts'],'replay',data['replay_error'],flush=True)
    for name,group in rows.items():
        print(name,{kind:round(values['teacher_kl'],5)
                    for kind,values in group.items()},flush=True)


if __name__=='__main__':main()

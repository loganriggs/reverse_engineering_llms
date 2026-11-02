"""Test whether a rotated quotient has reusable fixed-axis rule groups."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from eval_v3_rule_holdout_bond import HELD_OUT, panel, remove_duplicates
from eval_v3_two_path_holdout import prepare, readout
from surface_mixture_v3 import SurfaceMixtureV3


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir',default='language_task_suite/surface_mixture_v3_object8')
    p.add_argument('--basis-file',required=True)
    p.add_argument('--rotation-file',required=True)
    p.add_argument('--cal-seed',type=int,default=35001)
    p.add_argument('--test-seed',type=int,default=36001)
    p.add_argument('--cal-streams',type=int,default=96)
    p.add_argument('--test-streams',type=int,default=320)
    p.add_argument('--cal-cap',type=int,default=32)
    p.add_argument('--test-cap',type=int,default=160)
    p.add_argument('--length',type=int,default=72)
    p.add_argument('--group-size',type=int,default=4)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config
    folder=Path(a.model_dir)
    record=json.loads((folder/'results.json').read_text())['records'][0]
    model=BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(folder/f"model_depth{record['depth']}.pt",
                                     map_location='cpu',weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    kind,u=torch.load(a.basis_file,weights_only=True)['combined_teacher_optimized']
    assert kind=='combined'
    rotations=torch.load(a.rotation_file,weights_only=True)
    assert a.group_size<=u.shape[1]
    vocab=SurfaceMixtureV3(object_vocab_size=8).token2id
    seen=set()
    cal1,rem1=remove_duplicates(panel(a.cal_seed,a.cal_streams,a.cal_cap,a.length),seen)
    cal2,rem2=remove_duplicates(panel(a.cal_seed+1,a.cal_streams,a.cal_cap,a.length),seen)
    test,remtest=remove_duplicates(panel(a.test_seed,a.test_streams,a.test_cap,a.length),seen)
    for kind in HELD_OUT:
        assert sum(pair['kind']==kind for pair in cal1)>=a.cal_cap//2
        assert sum(pair['kind']==kind for pair in cal2)>=a.cal_cap//2
        assert sum(pair['kind']==kind for pair in test)>=a.test_cap//2
    c1=prepare(model,cal1,vocab)
    c2=prepare(model,cal2,vocab)
    te=prepare(model,test,vocab)
    results={}
    for name,rotation in rotations.items():
        axes=u@rotation
        co1=c1['dr']@axes
        co2=c2['dr']@axes
        cot=te['dr']@axes
        family={}
        for kind in HELD_OUT:
            m1=torch.tensor([label==kind for label in c1['kinds']])
            m2=torch.tensor([label==kind for label in c2['kinds']])
            mt=torch.tensor([label==kind for label in te['kinds']])
            selected1=co1[m1].square().mean(0).topk(a.group_size).indices
            selected2=co2[m2].square().mean(0).topk(a.group_size).indices
            group=torch.zeros(u.shape[1])
            group[selected1]=1
            delta=(cot*group)@axes.T
            effect=readout(model,te['r0']+delta)-te['p0']
            remaining=readout(model,te['r0']+te['dr']-delta)-te['p0']
            per_top=cot.abs().topk(a.group_size,dim=-1).indices
            per_mask=torch.zeros_like(cot).scatter_(1,per_top,1.0)
            per_delta=(cot*per_mask)@axes.T
            per_effect=readout(model,te['r0']+per_delta)-te['p0']
            energy=cot.square()
            family[kind]=dict(
                n_calibration_1=int(m1.sum()),n_calibration_2=int(m2.sum()),
                n_test=int(mt.sum()),
                selected_axes=sorted(int(j) for j in selected1),
                second_calibration_axes=sorted(int(j) for j in selected2),
                calibration_jaccard=(len(set(selected1.tolist())&set(selected2.tolist()))/
                                     len(set(selected1.tolist())|set(selected2.tolist()))),
                fixed_group_test_energy_share=(energy[mt][:,selected1].sum(-1)/
                                               energy[mt].sum(-1).clamp_min(1e-12)).mean().item(),
                fixed_group_to_teacher_l1=(effect[mt]-te['native'][mt]).abs().sum(-1).mean().item(),
                per_example_top_to_teacher_l1=(per_effect[mt]-te['native'][mt]).abs().sum(-1).mean().item(),
                full_subspace_to_teacher_l1=(
                    (readout(model,te['r0'][mt]+(cot[mt]@axes.T))-te['p0'][mt])-
                    te['native'][mt]).abs().sum(-1).mean().item(),
                remaining_after_group_l1=remaining[mt].abs().sum(-1).mean().item(),
                fixed_group_to_oracle_l1=(effect[mt]-te['oracle'][mt]).abs().sum(-1).mean().item(),
                native_to_oracle_l1=(te['native'][mt]-te['oracle'][mt]).abs().sum(-1).mean().item(),
                other_family_effect_l1={other:effect[torch.tensor(
                    [label==other for label in te['kinds']])].abs().sum(-1).mean().item()
                    for other in sorted(set(te['kinds'])) if other!=kind},
            )
        results[name]=family
    output=dict(config=vars(a),rank=u.shape[1],
                calibration_counts_1=dict(Counter(pair['kind'] for pair in cal1)),
                calibration_counts_2=dict(Counter(pair['kind'] for pair in cal2)),
                test_counts=dict(Counter(pair['kind'] for pair in test)),
                duplicates_removed=dict(calibration_1=rem1,calibration_2=rem2,test=remtest),
                max_endpoint_replay_error=max(c1['replay_error'],c2['replay_error'],te['replay_error']),
                results=results,
                caveat='Held-out family labels are used for the two small '
                       'axis-selection calibration panels. This tests fixed '
                       'group reuse, not unsupervised new-rule discovery. '
                       'Per-example top-k is an optimistic adaptive reference.')
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2)+'\n')
    for name,groups in results.items():
        print(name,{kind:{k:round(groups[kind][k],3) for k in
            ('calibration_jaccard','fixed_group_test_energy_share',
             'fixed_group_to_teacher_l1','per_example_top_to_teacher_l1')}
            for kind in HELD_OUT},flush=True)


if __name__=='__main__':
    main()

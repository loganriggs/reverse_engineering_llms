"""Fit a context-dependent TopK gate in a frozen v8 readout dictionary.

Fit/dev use section-key edits only. Relevant/irrelevant color edits and all
test prefixes are held out. This is a local featurizer experiment, not model
training or unseen-task discovery.
"""

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

from eval_v3_two_path_holdout import prepare, readout
from eval_v8_readout_sparse_gates import summarize, topk
from eval_v8_rule_holdout_readout import generator, scope_panel


class Gate(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(),
                                 nn.Linear(hidden, d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, coords):
        return coords.abs() + self.net(coords)


def masks(scores, k, temperature, straight_through):
    indices = scores.topk(k, dim=-1).indices
    hard = torch.zeros_like(scores).scatter_(-1, indices, 1.)
    assert torch.all(hard.sum(-1) == k)
    if not straight_through:
        return hard
    soft = torch.sigmoid(scores / temperature)
    return hard + soft - soft.detach()


def gated_delta(gate, data, center, u, k, temperature, straight_through):
    before = (data['r0']-center) @ u
    after = (data['r0']+data['dr']-center) @ u
    before_mask = masks(gate(before), k, temperature, straight_through)
    after_mask = masks(gate(after), k, temperature, straight_through)
    delta = ((after*after_mask)-(before*before_mask)) @ u.T
    return delta, before_mask > .5, after_mask > .5


def loss_on(model, gate, data, center, u, k, temperature):
    delta, _, _ = gated_delta(gate, data, center, u, k, temperature, True)
    effect = readout(model, data['r0']+delta)-data['p0']
    return (effect-data['native']).square().sum(-1).mean()


@torch.no_grad()
def eval_on(model, gate, data, center, u, k, temperature):
    delta, m0, m1 = gated_delta(gate, data, center, u, k, temperature, False)
    return summarize(model, data, delta, m0, m1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir',default='language_task_suite/results/v8_scope_balanced_continue_seed101')
    p.add_argument('--basis',default='language_task_suite/results/v8_scope_keyfit_colorholdout_rank16_seed46001.pt')
    p.add_argument('--fit-seed',type=int,default=46001)
    p.add_argument('--dev-seed',type=int,default=46002)
    p.add_argument('--test-seeds',default='48001,48002')
    p.add_argument('--fit-streams',type=int,default=500)
    p.add_argument('--dev-streams',type=int,default=300)
    p.add_argument('--test-streams',type=int,default=500)
    p.add_argument('--cap',type=int,default=120)
    p.add_argument('--k',type=int,default=4)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--temperature',type=float,default=.2)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config
    saved=json.loads((Path(a.model_dir)/'results.json').read_text())
    assert saved['config']['generator']=='v8'
    record=saved['records'][0]
    model=BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(Path(a.model_dir)/f"model_depth{record['depth']}.pt",
                                     map_location='cpu',weights_only=True))
    model.eval()
    for param in model.parameters(): param.requires_grad_(False)
    kwargs=saved['generator_kwargs']
    vocab=generator(0,kwargs).token2id
    kind,u=torch.load(a.basis,map_location='cpu',weights_only=True)['fit_delta_pca']
    assert kind=='combined' and 0<a.k<=u.shape[1]

    fit_raw,_=scope_panel(a.fit_seed,kwargs,a.fit_streams,a.cap)
    dev_raw,_=scope_panel(a.dev_seed,kwargs,a.dev_streams,a.cap)
    fit_pairs=[x for x in fit_raw if x['kind']=='section_key']
    dev_pairs=[x for x in dev_raw if x['kind']=='section_key']
    assert fit_pairs and dev_pairs
    fit=prepare(model,fit_pairs,vocab)
    dev=prepare(model,dev_pairs,vocab)
    center=torch.cat((fit['r0'],fit['r0']+fit['dr'])).mean(0)
    torch.manual_seed(a.seed)
    gate=Gate(u.shape[1],a.hidden)
    optimizer=torch.optim.Adam(gate.parameters(),lr=a.lr)
    best=(float('inf'),0,copy.deepcopy(gate.state_dict()))
    records=[]
    initial_gradient_norm=None
    for step in range(a.steps+1):
        if step%10==0 or step==a.steps:
            with torch.no_grad():
                dev_loss=float(loss_on(model,gate,dev,center,u,a.k,a.temperature))
            if dev_loss<best[0]:
                best=(dev_loss,step,copy.deepcopy(gate.state_dict()))
            records.append(dict(step=step,dev_squared_effect=dev_loss))
        if step==a.steps: break
        optimizer.zero_grad()
        loss=loss_on(model,gate,fit,center,u,a.k,a.temperature)
        loss.backward()
        grad_norm=float(torch.nn.utils.clip_grad_norm_(gate.parameters(),10.))
        if step==0:
            initial_gradient_norm=grad_norm
            assert initial_gradient_norm>0 and torch.isfinite(torch.tensor(grad_norm))
        optimizer.step()
    gate.load_state_dict(best[2])
    test={}
    for seed in (int(x) for x in a.test_seeds.split(',')):
        raw,_=scope_panel(seed,kwargs,a.test_streams,a.cap)
        assert set(x['kind'] for x in raw)=={
            'section_key','relevant_color','irrelevant_color'}
        data=prepare(model,raw,vocab)
        before=(data['r0']-center)@u
        after=(data['r0']+data['dr']-center)@u
        b0,b1=topk(before,a.k),topk(after,a.k)
        baseline=summarize(model,data,(b1-b0)@u.T,b0!=0,b1!=0)
        test[str(seed)]=dict(counts=dict(Counter(x['kind'] for x in raw)),
                             replay_error=data['replay_error'],
                             state_topk=baseline,
                             learned_gate=eval_on(model,gate,data,center,u,
                                                  a.k,a.temperature))
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    checkpoint=path.with_suffix('.pt')
    torch.save(dict(gate=gate.state_dict(),dictionary=u,center=center,
                    selected_step=best[1]),checkpoint)
    result=dict(config=vars(a),checkpoint=str(checkpoint),
                fit_count=len(fit_pairs),dev_count=len(dev_pairs),
                max_fit_dev_replay_error=max(fit['replay_error'],dev['replay_error']),
                initial_gradient_norm=initial_gradient_norm,
                best_dev_squared_effect=best[0],best_step=best[1],
                learning_curve=records,test=test,
                caveat='Straight-through TopK has a surrogate gradient; only '
                       'the hard gate is used at evaluation. Fit/dev use key '
                       'edits only; color edits and test prefixes are held out. '
                       'This is local teacher-effect fidelity, not a proven '
                       'semantic group or unseen-task discovery.')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('best_step',best[1],'initial_grad',initial_gradient_norm,flush=True)
    for seed,rows in test.items():
        print(seed,{name:round(item['relevant_color']['sufficient_to_teacher_l1'],4)
                    for name,item in rows.items() if name in ('state_topk','learned_gate')},
              flush=True)


if __name__=='__main__':main()

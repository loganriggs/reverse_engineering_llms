"""Compare sparse readout featurizers on v8 whole association holdouts.

Fit/dev use only section-key edits whose full fact histories exclude eight
held-out (section ID,color) associations. Test both seen and heldout
associations, including key/relevant/irrelevant-color edit effects. The
language model is frozen and was trained on every association.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_v3_rule_holdout_bond import capture as capture_final_bond
from eval_v3_two_path_holdout import prepare as prepare_final_residual, readout
from surface_mixture_v8 import SurfaceMixtureV8
from train_surface_v2_shared_groups import ScalarTopK, initialize


def normalized_data(raw,center,scale):
    before=(raw['r0']-center)/scale
    after=(raw['r0']+raw['dr']-center)/scale
    return before,after


def reconstruct(feature,x,center,scale,membership=None):
    result=(feature(x,membership=membership) if membership is not None
            else feature(x))
    return result['reconstruction']*scale+center,result['support']


def output_prob(model,data,feature_state,endpoint,site,indices=slice(None)):
    if site=='final_residual':
        return readout(model,feature_state)
    if site=='final_bond':
        context=data[f'h{endpoint}'][indices]
        output=model.blocks[-1].attn.out
        return readout(model,context+F.linear(feature_state,output.weight))
    assert site in ('layer0_bond','layer1_bond')
    layer=int(site[5])
    state=data[f'ctx{endpoint}'][indices].clone()
    last=data[f'last{endpoint}'][indices]
    n=len(state)
    native=(data['r0'] if endpoint==0 else data['r0']+data['dr'])[indices]
    change=F.linear(feature_state-native,
                    model.blocks[layer].attn.out.weight)
    state[torch.arange(n,device=state.device),last]+=change
    for block in model.blocks[layer+1:]:
        state=block(state)
    if model.use_rmsnorm or model.use_final_norm:
        state=model.final_norm(state)
    logits=model.lm_head(state)
    return logits[torch.arange(n,device=state.device),last].softmax(-1)


@torch.no_grad()
def prepare_bond(model,pairs,vocab):
    h0,z0,p0=capture_final_bond(model,[p['original'] for p in pairs])
    h1,z1,p1=capture_final_bond(model,[p['edited'] for p in pairs])
    data=dict(r0=z0,dr=z1-z0,h0=h0,h1=h1,p0=p0,
              native=p1-p0,kinds=[p['kind'] for p in pairs])
    data['oracle']=torch.zeros_like(p0)
    for i,pair in enumerate(pairs):
        for token,mass in pair['q_after'].items():
            data['oracle'][i,vocab[token]]+=mass
        for token,mass in pair['q_before'].items():
            data['oracle'][i,vocab[token]]-=mass
    replay=max((output_prob(model,data,z0,0,'final_bond')-p0).abs().max(),
               (output_prob(model,data,z1,1,'final_bond')-p1).abs().max())
    data['replay_error']=float(replay)
    if data['replay_error']>1e-5:
        raise AssertionError(f'Final-bond endpoint replay {replay}')
    return data


@torch.no_grad()
def capture_early(model,prefixes,layer,pad_id,batch_size=32):
    device=model.embed.weight.device
    width=model.config.n_embd
    length=model.config.seq_len
    contexts=torch.empty((len(prefixes),length,width),device=device)
    bond=torch.empty((len(prefixes),width),device=device)
    probs=torch.empty((len(prefixes),model.config.vocab_size),device=device)
    last=torch.empty(len(prefixes),dtype=torch.long,device=device)
    block=model.blocks[layer]
    for start in range(0,len(prefixes),batch_size):
        group=prefixes[start:start+batch_size]
        maxlen=max(map(len,group))
        ids=torch.full((len(group),maxlen),pad_id,dtype=torch.long,device=device)
        positions=torch.tensor([len(p)-1 for p in group],device=device)
        for i,prefix in enumerate(group):
            ids[i,:len(prefix)]=torch.tensor(prefix,device=device)
        captured={}
        def z_hook(_module,inputs):
            captured['z']=inputs[0].detach()
        def block_hook(_module,_inputs,output):
            captured['ctx']=output.detach()
        hz=block.attn.out.register_forward_pre_hook(z_hook)
        hb=block.register_forward_hook(block_hook)
        try:
            logits=model(ids)[0]
        finally:
            hb.remove();hz.remove()
        j=torch.arange(len(group),device=device)
        contexts[start:start+len(group)].zero_()
        contexts[start:start+len(group),:maxlen]=captured['ctx']
        bond[start:start+len(group)]=captured['z'][j,positions]
        probs[start:start+len(group)]=logits[j,positions].softmax(-1)
        last[start:start+len(group)]=positions
    # Causal padding must not alter valid-position probabilities.
    for i in range(min(3,len(prefixes))):
        raw=model(torch.tensor(prefixes[i],device=device)[None])[0][0,-1]
        error=(raw.softmax(-1)-probs[i]).abs().max()
        if error>1e-5:
            raise AssertionError(f'Right-padding changed answer {i}: {error}')
    return contexts,bond,probs,last


@torch.no_grad()
def prepare_early(model,pairs,vocab,layer,pad_id):
    prefixes0=[pair['original'] for pair in pairs]
    prefixes1=[pair['edited'] for pair in pairs]
    ctx0,z0,p0,last0=capture_early(model,prefixes0,layer,pad_id)
    ctx1,z1,p1,last1=capture_early(model,prefixes1,layer,pad_id)
    data=dict(r0=z0,dr=z1-z0,ctx0=ctx0,ctx1=ctx1,
              last0=last0,last1=last1,p0=p0,native=p1-p0,
              kinds=[pair['kind'] for pair in pairs])
    data['oracle']=torch.zeros_like(p0)
    for i,pair in enumerate(pairs):
        for token,mass in pair['q_after'].items():
            data['oracle'][i,vocab[token]]+=mass
        for token,mass in pair['q_before'].items():
            data['oracle'][i,vocab[token]]-=mass
    site=f'layer{layer}_bond'
    replay=max((output_prob(model,data,z0,0,site)-p0).abs().max(),
               (output_prob(model,data,z1,1,site)-p1).abs().max())
    data['replay_error']=float(replay)
    if replay>1e-5:
        raise AssertionError(f'Early-bond endpoint replay {replay}')
    return data


def loss_on(model,feature,data,center,scale,indices,causal_weight,site,
            membership=None):
    x0,x1=normalized_data(data,center,scale)
    x0,x1=x0[indices],x1[indices]
    r0,s0=reconstruct(feature,x0,center,scale,membership)
    r1,s1=reconstruct(feature,x1,center,scale,membership)
    target0=data['r0'][indices]
    target1=target0+data['dr'][indices]
    mse=(F.mse_loss(r0,target0)+F.mse_loss(r1,target1))/(scale**2)
    if causal_weight or isinstance(indices,slice):
        predicted=(output_prob(model,data,r1,1,site,indices)-
                   output_prob(model,data,r0,0,site,indices))
        effect=(predicted-data['native'][indices]).square().sum(-1).mean()
    else:
        effect=r0.new_zeros(())
    return mse+causal_weight*effect,dict(mse=mse,effect_mse=effect,
                                         active=(s0.sum(-1).mean()+
                                                 s1.sum(-1).mean())/2)


def fit(model,feature,train,dev,center,scale,steps,causal_weight,site,
        seed,membership=None):
    rng=torch.Generator().manual_seed(seed)
    optimizer=torch.optim.AdamW((p for p in feature.parameters()
                                 if p.requires_grad),lr=.002,weight_decay=.0001)
    best=(float('inf'),0,copy.deepcopy(feature.state_dict()))
    records=[]
    for step in range(steps+1):
        if step==0 or step%50==0 or step==steps:
            feature.eval()
            with torch.no_grad():
                objective,parts=loss_on(model,feature,dev,center,scale,
                                        slice(None),causal_weight,site,membership)
                score=float(objective)
                if score<best[0]:
                    best=(score,step,copy.deepcopy(feature.state_dict()))
                records.append(dict(step=step,objective=score,
                                    reconstruction_mse=float(parts['mse']),
                                    teacher_effect_mse=float(parts['effect_mse']),
                                    active_atoms=float(parts['active'])))
        if step==steps:break
        feature.train()
        indices=torch.randint(len(train['r0']),(64,),generator=rng).to(
            train['r0'].device)
        objective,_=loss_on(model,feature,train,center,scale,indices,
                            causal_weight,site,membership)
        if not torch.isfinite(objective):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(feature.parameters(),10.)
        optimizer.step()
    feature.load_state_dict(best[2])
    feature.eval()
    return dict(best_dev_objective=best[0],best_step=best[1],curve=records)


def topk_basis(x,basis,k):
    coordinates=x@basis
    support=torch.zeros_like(coordinates).scatter_(
        -1,coordinates.abs().topk(k,-1).indices,1.)
    return (coordinates*support)@basis.T,support


@torch.no_grad()
def evaluate(model,data,center,scale,feature=None,membership=None,
             basis=None,k=None,site='final_residual'):
    x0,x1=normalized_data(data,center,scale)
    if feature is not None:
        r0,s0=reconstruct(feature,x0,center,scale,membership)
        r1,s1=reconstruct(feature,x1,center,scale,membership)
    elif basis is not None:
        a0,s0=topk_basis(x0,basis,k)
        a1,s1=topk_basis(x1,basis,k)
        r0,r1=a0*scale+center,a1*scale+center
    else:
        r0,r1=data['r0'],data['r0']+data['dr']
        s0=s1=torch.ones_like(x0)
    p0=output_prob(model,data,r0,0,site)
    p1=output_prob(model,data,r1,1,site)
    native0=data['p0']
    native1=native0+data['native']
    native_log0=native0.clamp_min(1e-30).log()
    native_log1=native1.clamp_min(1e-30).log()
    kl0=(native0*(native_log0-p0.clamp_min(1e-30).log())).sum(-1)
    kl1=(native1*(native_log1-p1.clamp_min(1e-30).log())).sum(-1)
    teacher_kl=((kl0+kl1)/2).clamp_min(0)
    effect=(p1-p0)
    target0=data['oracle_before']
    target1=data['oracle_after']
    correct=((p0.argmax(-1)==target0).float()+
             (p1.argmax(-1)==target1).float())/2
    ix=torch.arange(len(p0),device=p0.device)
    ce=(-p0[ix,target0].clamp_min(1e-30).log()
        -p1[ix,target1].clamp_min(1e-30).log())/2
    rows={}
    for kind in sorted(set(data['kinds'])):
        mask=torch.tensor([label==kind for label in data['kinds']],
                          device=p0.device)
        l1=lambda x:float(x[mask].abs().sum(-1).mean())
        rows[kind]=dict(n=int(mask.sum()),
                        relative_residual_mse=float(
                            ((r0[mask]-data['r0'][mask]).square().sum(-1)+
                             (r1[mask]-data['r0'][mask]-data['dr'][mask])
                              .square().sum(-1)).sum()/
                            ((data['r0'][mask]-center).square().sum()+
                             (data['r0'][mask]+data['dr'][mask]-center)
                              .square().sum()).clamp_min(1e-30)),
                        teacher_kl=float(teacher_kl[mask].mean()),
                        answer_accuracy=float(correct[mask].mean()),
                        answer_ce=float(ce[mask].mean()),
                        native_effect_l1=l1(data['native']),
                        effect_to_teacher_l1=l1(effect-data['native']),
                        effect_to_oracle_l1=l1(effect-data['oracle']),
                        active_atoms=float(((s0[mask].sum(-1)+
                                             s1[mask].sum(-1))/2).mean()))
    return rows


def add_targets(data,pairs,vocab):
    before=[]
    after=[]
    for pair in pairs:
        for field,storage in (('q_before',before),('q_after',after)):
            q=pair[field]
            assert len(q)==1 and next(iter(q.values()))==1.
            storage.append(vocab[next(iter(q))])
    data['oracle_before']=torch.tensor(before,device=data['r0'].device)
    data['oracle_after']=torch.tensor(after,device=data['r0'].device)
    return data


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--feature-source',default='/home/loganriggs/Coding/neuron_circuits/scripts')
    p.add_argument('--model-dir',required=True)
    p.add_argument('--panels',required=True)
    p.add_argument('--site',choices=('final_residual','final_bond',
                                    'layer0_bond','layer1_bond'),
                   default='final_residual')
    p.add_argument('--device',default='cpu')
    p.add_argument('--baselines-only',action='store_true',
                   help='Evaluate native and weight-only baselines without fitting models')
    p.add_argument('--models',default=None,
                   help='Comma-separated subset of learned candidate names')
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--causal-weight',type=float,default=1.)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--threads',type=int,default=1)
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
    if device.type!='cpu' and a.site in ('final_residual','final_bond'):
        p.error('GPU mode currently supports layer0_bond/layer1_bond only')
    model=BilinearGPT(Config(**record['model_config'])).to(device)
    model.load_state_dict(torch.load(folder/f"model_depth{record['depth']}.pt",
                                     map_location=device,weights_only=True))
    model.eval().requires_grad_(False)
    assert model.config.n_embd==64
    if a.site=='final_bond':
        assert model.blocks[-1].mlp is None
    if a.site in ('layer0_bond','layer1_bond'):
        assert model.blocks[int(a.site[5])].mlp is None
    panel_file=json.loads(Path(a.panels).read_text())
    assert len(panel_file['heldout_universe'])==8
    vocab=SurfaceMixtureV8(seed=0,**saved['generator_kwargs']).token2id
    data={}
    for name,rows in panel_file['panels'].items():
        pairs=[row['pair'] for row in rows]
        if a.site=='final_residual':
            raw=prepare_final_residual(model,pairs,vocab)
        elif a.site=='final_bond':
            raw=prepare_bond(model,pairs,vocab)
        else:
            raw=prepare_early(model,pairs,vocab,int(a.site[5]),vocab['.'])
        data[name]=add_targets(raw,pairs,vocab)
        print(name,len(pairs),'replay',data[name]['replay_error'],flush=True)
    center=torch.cat((data['fit']['r0'],
                      data['fit']['r0']+data['fit']['dr'])).mean(0)
    scale=torch.cat((data['fit']['r0']-center,
                     data['fit']['r0']+data['fit']['dr']-center)).square().mean()
    scale=scale.sqrt().clamp_min(1e-6)
    x0,x1=normalized_data(data['fit'],center,scale)
    _,_,vh=torch.linalg.svd(torch.cat((x0,x1)).double(),full_matrices=False)
    directions=vh.T.float()
    if directions.shape[1]<64:
        directions=torch.linalg.qr(torch.cat((directions,torch.eye(64,device=device)),
                                             dim=1)).Q
    fixed=torch.zeros(16,64,device=device)
    for atom in range(64):fixed[atom%16,atom]=1.
    torch.manual_seed(a.seed)
    models={
        'scalar16_reconstruction':ScalarTopK(64,64,16),
        'scalar16_causal':ScalarTopK(64,64,16),
        'disjoint16_causal':SharedGroups(64,64,16,4,4),
        'shared16_reconstruction':SharedGroups(64,64,16,4,4),
        'shared16_causal':SharedGroups(64,64,16,4,4),
        'scalar8_reconstruction':ScalarTopK(64,64,8),
        'scalar8_causal':ScalarTopK(64,64,8),
        'scalar10_reconstruction':ScalarTopK(64,64,10),
        'scalar10_causal':ScalarTopK(64,64,10),
    }
    if a.models:
        chosen=a.models.split(',')
        if any(name not in models for name in chosen):
            p.error(f'Unknown candidate in --models: {chosen}')
        models={name:models[name] for name in chosen}
    models={name:feature.to(device) for name,feature in models.items()}
    memberships={'disjoint16_causal':fixed}
    histories={}
    if a.baselines_only:
        models={}
    else:
        for name,feature in models.items():
            initialize(feature,directions)
            if name in memberships:
                feature.membership_logits.requires_grad_(False)
            weight=a.causal_weight if name.endswith('_causal') else 0.
            histories[name]=fit(model,feature,data['fit'],data['dev'],
                                center,scale,a.steps,weight,a.site,a.seed+101,
                                memberships.get(name))
            print(name,histories[name]['best_step'],
                  histories[name]['best_dev_objective'],flush=True)
    weight=(model.lm_head.weight if a.site=='final_residual'
            else model.blocks[-1 if a.site=='final_bond'
                              else int(a.site[5])].attn.out.weight)
    _,_,weight_vh=torch.linalg.svd(weight.double(),
                                   full_matrices=False)
    baselines={'physical16':(torch.eye(64,device=device),16),
               'weight_svd16':(weight_vh.T.float(),16)}
    if a.site in ('layer0_bond','layer1_bond'):
        for rank in (8,10):
            baselines[f'weight_svd{rank}']=(weight_vh.T.float(),rank)
    if a.site=='final_bond':
        composed=model.lm_head.weight.detach() @ model.blocks[-1].attn.out.weight.detach()
        _,_,composed_vh=torch.linalg.svd(composed.double(),full_matrices=False)
        for rank in (8,10,16):
            baselines[f'composed_logit_svd{rank}']=(composed_vh.T.float(),rank)
    panels={}
    for panel in ('test_seen','test_heldout'):
        rows={'native':evaluate(model,data[panel],center,scale,site=a.site)}
        for name,(basis,budget) in baselines.items():
            rows[name]=evaluate(model,data[panel],center,scale,basis=basis,k=budget,
                                site=a.site)
        for name,feature in models.items():
            rows[name]=evaluate(model,data[panel],center,scale,feature,
                                memberships.get(name),site=a.site)
        panels[panel]=rows
        print(panel,{name:round(values['section_key']['effect_to_teacher_l1'],3)
                     for name,values in rows.items()},flush=True)
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    checkpoint=None
    if models:
        checkpoint=path.with_suffix('.pt')
        torch.save(dict(center=center,scale=scale,
                        states={name:feature.state_dict()
                                for name,feature in models.items()},
                        fixed_membership=fixed),checkpoint)
    result=dict(config=vars(a),checkpoint=str(checkpoint) if checkpoint else None,
                panel_summary=panel_file['summary'],
                max_endpoint_replay_error=max(row['replay_error']
                                              for row in data.values()),
                histories=histories,panels=panels,
                caveat='Whole association holdout applies to featurizer fit/dev '
                       'only. The frozen teacher learned all combinations. '
                       'Fit/dev use key edits; test additionally uses color edits. '
                       'This is local fidelity at the configured site, not full-model '
                       'compression or new-task discovery.')
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

"""Ablate semantic attention edges in trained v8 section models.

Each intervention subtracts the selected unnormalized bilinear attention
contributions after W_O at one layer. It leaves all other query/key edges
and layers unchanged. Evaluate supported section/key/color edit effects.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_surface_mixture_v7_scope import build as scope_build
from surface_mixture_v2 import _split
from surface_mixture_v4 import _content
from surface_mixture_v8 import SurfaceMixtureV8


PATHS=(
    'answer_to_key',
    'answer_to_target_header',
    'answer_to_other_headers',
    'answer_to_target_color',
    'answer_to_other_colors',
    'answer_to_all_headers',
    'answer_to_all_colors',
    'answer_to_recent16',
    'key_to_target_header',
    'key_to_other_headers',
    'key_to_all_headers',
    'key_to_recent16',
    'target_color_to_target_header',
)


def semantic_edges(words, path):
    history,raw=_split(words)
    clause=_content(raw)
    assert clause[:7]==['What','color','was','the','badge','in','section']
    answer=len(words)-1
    key=answer-3
    target=clause[7]
    assert words[key]==target
    headers={}
    colors={}
    active=None
    offset=0
    for line in history:
        content=_content(line)
        start=offset+len(line)-len(content)
        if content and content[0]=='Section':
            active=content[1]
            headers[active]=start+1
        elif content and content[0]=='Detail':
            assert active is not None
            colors[active]=start+5
            active=None
        offset+=len(line)
    assert target in headers and target in colors
    assert all(words[index]==section for section,index in headers.items())
    other_headers=[p for name,p in headers.items() if name!=target]
    other_colors=[p for name,p in colors.items() if name!=target]
    routes={
        'answer_to_key':[(answer,key)],
        'answer_to_target_header':[(answer,headers[target])],
        'answer_to_other_headers':[(answer,p) for p in other_headers],
        'answer_to_target_color':[(answer,colors[target])],
        'answer_to_other_colors':[(answer,p) for p in other_colors],
        'answer_to_all_headers':[(answer,p) for p in headers.values()],
        'answer_to_all_colors':[(answer,p) for p in colors.values()],
        'answer_to_recent16':[(answer,p) for p in
                              range(max(0,answer-15),answer+1)],
        'key_to_target_header':[(key,headers[target])],
        'key_to_other_headers':[(key,p) for p in other_headers],
        'key_to_all_headers':[(key,p) for p in headers.values()],
        'key_to_recent16':[(key,p) for p in range(max(0,key-15),key+1)],
        'target_color_to_target_header':[(colors[target],headers[target])],
    }
    assert all(q>=k for edges in routes.values() for q,k in edges)
    return routes[path]


def prepare_prefixes(pairs, gen):
    prefixes=[]
    index={}
    for pair in pairs:
        for field in ('original','edited'):
            key=tuple(pair[field])
            if key not in index:
                index[key]=len(prefixes)
                prefixes.append(list(key))
    edges={path:[semantic_edges([gen.id2token[i] for i in prefix],path)
                 for prefix in prefixes] for path in PATHS}
    before=torch.tensor([index[tuple(p['original'])] for p in pairs])
    after=torch.tensor([index[tuple(p['edited'])] for p in pairs])
    return prefixes,edges,before,after


def pad_prefixes(prefixes, pad_id, device):
    length=max(map(len,prefixes))
    ids=torch.full((len(prefixes),length),pad_id,dtype=torch.long,device=device)
    last=[]
    for b,prefix in enumerate(prefixes):
        ids[b,:len(prefix)]=torch.tensor(prefix,device=device)
        last.append(len(prefix)-1)
    return ids,torch.tensor(last,device=device)


@torch.no_grad()
def edge_bond(attn,x,edge_lists,apply_rotary_emb):
    """Pre-output bond contribution for a ragged batch of (query,key) edges."""
    batch,length,width=x.shape
    heads,head_dim=attn.n_head,attn.head_dim
    max_edges=max(map(len,edge_lists),default=0)
    bond=x.new_zeros((batch,length,width))
    if not max_edges:
        return bond
    qpos=torch.zeros((batch,max_edges),dtype=torch.long,device=x.device)
    kpos=torch.zeros_like(qpos)
    keep=torch.zeros_like(qpos,dtype=torch.bool)
    for b,edges in enumerate(edge_lists):
        for e,(q,k) in enumerate(edges):
            assert 0<=k<=q<length
            qpos[b,e],kpos[b,e],keep[b,e]=q,k,True
    shape=(batch,length,heads,head_dim)
    q1,k1,q2,k2=(projection(x).view(shape) for projection in
                  (attn.q1,attn.k1,attn.q2,attn.k2))
    v=attn.v(x).view(shape)
    cos,sin=attn.rotary(q1)
    q1,k1=apply_rotary_emb(q1,cos,sin),apply_rotary_emb(k1,cos,sin)
    q2,k2=apply_rotary_emb(q2,cos,sin),apply_rotary_emb(k2,cos,sin)
    if attn.qk_norm:
        q1,k1,q2,k2=(F.normalize(z,dim=-1) for z in (q1,k1,q2,k2))
    bi=torch.arange(batch,device=x.device)[:,None]
    first=(q1[bi,qpos]*k1[bi,kpos]).sum(-1)/head_dim
    second=(q2[bi,qpos]*k2[bi,kpos]).sum(-1)/head_dim
    contribution=(first*second)[...,None]*v[bi,kpos]*keep[...,None,None]
    for e in range(max_edges):
        bond[bi[:,0],qpos[:,e]]+=contribution[:,e].reshape(batch,width)
    return bond


@torch.no_grad()
def edge_delta(attn,x,edge_lists,apply_rotary_emb):
    """Negative W_O output for a ragged batch of (query,key) edges."""
    return -F.linear(edge_bond(attn,x,edge_lists,apply_rotary_emb),
                     attn.out.weight)


@torch.no_grad()
def predict(model,prefixes,edge_lists,layer,apply_rotary_emb,pad_id,
            device,batch_size,operation='remove'):
    """Predict with one layer or a dict of layer→edge-list interventions."""
    assert operation in ('remove','keep')
    outputs=[]
    for start in range(0,len(prefixes),batch_size):
        group=prefixes[start:start+batch_size]
        ids,last=pad_prefixes(group,pad_id,device)
        if layer is None:
            logits=model(ids)[0]
        else:
            interventions=({layer:edge_lists} if isinstance(layer,int)
                           else layer)
            hooks=[]
            def install(attn,selected):
                state={}
                def capture(_module,inputs):
                    state['delta']=edge_delta(attn,inputs[0],selected,
                                              apply_rotary_emb)
                def subtract(_module,_inputs,output):
                    delta=state.pop('delta')
                    if operation=='remove':
                        return output+delta
                    replaced=output.clone()
                    for b,edges in enumerate(selected):
                        for q in {q for q,_ in edges}:
                            replaced[b,q]=-delta[b,q]
                    return replaced
                hooks.append(attn.register_forward_pre_hook(capture))
                hooks.append(attn.out.register_forward_hook(subtract))
            for intervention_layer,all_edges in interventions.items():
                install(model.blocks[intervention_layer].attn,
                        all_edges[start:start+batch_size])
            try:
                logits=model(ids)[0]
            finally:
                for hook in reversed(hooks):
                    hook.remove()
        outputs.append(logits[torch.arange(len(group),device=device),last]
                       .softmax(-1).cpu())
    return torch.cat(outputs)


def scores(pairs,baseline,patched,before,after,gen):
    native=baseline[after]-baseline[before]
    changed=patched[after]-patched[before]
    oracle=torch.zeros_like(native)
    for i,pair in enumerate(pairs):
        for token,mass in pair['q_after'].items():
            oracle[i,gen.token2id[token]]+=mass
        for token,mass in pair['q_before'].items():
            oracle[i,gen.token2id[token]]-=mass
    groups={}
    answer_correct=[]
    answer_ce=[]
    for i,pair in enumerate(pairs):
        endpoints=[]
        for field,index in (('q_before',before[i]),('q_after',after[i])):
            distribution=pair[field]
            assert len(distribution)==1 and next(iter(distribution.values()))==1.
            token=gen.token2id[next(iter(distribution))]
            prob=patched[index]
            endpoints.append((int(prob.argmax()==token),
                              -float(prob[token].clamp_min(1e-30).log())))
        answer_correct.append(sum(x[0] for x in endpoints)/2)
        answer_ce.append(sum(x[1] for x in endpoints)/2)
    for kind in sorted({p['kind'] for p in pairs}):
        mask=torch.tensor([p['kind']==kind for p in pairs])
        l1=lambda x:float(x[mask].abs().sum(-1).mean())
        kl=(baseline* (baseline.clamp_min(1e-30).log()-
                       patched.clamp_min(1e-30).log())).sum(-1).clamp_min(0)
        endpoint=torch.unique(torch.cat((before[mask],after[mask])))
        groups[kind]=dict(n=int(mask.sum()),
                          answer_accuracy=float(torch.tensor(answer_correct)[mask].mean()),
                          answer_ce=float(torch.tensor(answer_ce)[mask].mean()),
                          native_effect_l1=l1(native),
                          oracle_effect_l1=l1(oracle),
                          effect_to_teacher_l1=l1(changed-native),
                          effect_to_oracle_l1=l1(changed-oracle),
                          mean_endpoint_teacher_kl=float(kl[endpoint].mean()),
                          mean_endpoint_prediction_l1=float(
                              (patched[endpoint]-baseline[endpoint])
                              .abs().sum(-1).mean()))
    return groups


def parse_circuit(spec,edge_map,n_layers):
    """Parse `0:path+path,1:path,2:path` into batched edge lists."""
    circuit={}
    for stage in spec.split(','):
        layer_text,path_text=stage.split(':',1)
        layer=int(layer_text)
        assert 0<=layer<n_layers and layer not in circuit
        names=path_text.split('+')
        assert names and all(name in edge_map for name in names)
        circuit[layer]=[sorted(set(edge for name in names
                                   for edge in edge_map[name][i]))
                        for i in range(len(next(iter(edge_map.values()))))]
    assert circuit
    return circuit


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir',required=True)
    p.add_argument('--streams',type=int,default=300)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--key-cap',type=int,default=50)
    p.add_argument('--seed',type=int,default=74001)
    p.add_argument('--paths',default=','.join(PATHS))
    p.add_argument('--layers',default='0,1,2')
    p.add_argument('--operation',choices=('remove','keep'),default='remove')
    p.add_argument('--circuit',default=None,
                   help='e.g. 0:key_to_all_headers,1:answer_to_all_colors,2:answer_to_key')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--torch-threads',type=int,default=1)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(a.torch_threads)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config,apply_rotary_emb
    folder=Path(a.model_dir)
    saved=json.loads((folder/'results.json').read_text())
    record=saved['records'][0]
    model=BilinearGPT(Config(**record['model_config'])).to(a.device)
    model.load_state_dict(torch.load(folder/f"model_depth{record['depth']}.pt",
                                     map_location=a.device,weights_only=True))
    model.eval()
    gen=SurfaceMixtureV8(seed=a.seed,**saved['generator_kwargs'])
    _,pairs,attempted=scope_build(gen,a.streams,a.length,a.key_cap)
    assert pairs and {p['kind'] for p in pairs}=={
        'section_key','relevant_color','irrelevant_color'}
    prefixes,edge_map,before,after=prepare_prefixes(pairs,gen)
    paths=a.paths.split(',')
    layers=[int(x) for x in a.layers.split(',')]
    assert all(path in PATHS for path in paths)
    assert all(0<=layer<len(model.blocks) for layer in layers)
    pad_id=gen.token2id['.']
    baseline=predict(model,prefixes,None,None,apply_rotary_emb,pad_id,
                     a.device,a.batch_size)
    baseline_scores=scores(pairs,baseline,baseline,before,after,gen)
    # A empty-edge hook must replay the unmodified model exactly.
    identity=predict(model,prefixes,[[] for _ in prefixes],layers[0],
                     apply_rotary_emb,pad_id,a.device,a.batch_size)
    identity_error=float((identity-baseline).abs().max())
    assert identity_error<1e-7
    result={}
    if a.circuit:
        circuit=parse_circuit(a.circuit,edge_map,len(model.blocks))
        patched=predict(model,prefixes,None,circuit,apply_rotary_emb,pad_id,
                        a.device,a.batch_size,a.operation)
        result['circuit']=scores(pairs,baseline,patched,before,after,gen)
        print('circuit',a.circuit,'key accuracy',
              result['circuit']['section_key']['answer_accuracy'],'key effect',
              result['circuit']['section_key']['effect_to_teacher_l1'],flush=True)
    else:
        for layer in layers:
            for path in paths:
                patched=predict(model,prefixes,edge_map[path],layer,
                                apply_rotary_emb,pad_id,a.device,a.batch_size,
                                a.operation)
                row=scores(pairs,baseline,patched,before,after,gen)
                result[f'layer{layer}_{path}']=row
                print(layer,path,'key effect',round(
                    row['section_key']['effect_to_teacher_l1'],5),'color effect',
                    round(row['relevant_color']['effect_to_teacher_l1'],5),
                    flush=True)
    output=dict(config=vars(a),n_unique_prefixes=len(prefixes),
                pair_counts=dict(Counter(p['kind'] for p in pairs)),
                attempted=dict(attempted),identity_max_probability_error=identity_error,
                baseline=baseline_scores,results=result,
                caveat='The operation removes or keeps one semantic edge family '
                       'at its query positions in one attention layer. A keep '
                       'test alone does not prove a sufficient multi-layer '
                       'circuit or a unique algorithm. '
                       'The pairs pass exact full-prefix generator support.')
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()

"""Exact selected-anchor attention paths for all 16 crossed skip rules."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

_pre=argparse.ArgumentParser(add_help=False)
_pre.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
_source,_=_pre.parse_known_args()
sys.path.insert(0,_source.source)

from model import BilinearGPT, Config, apply_rotary_emb
from original_v3_crossed_skip import LanguageV3CrossedSkip, nearest_skip_anchor
from train_original_v3_shared_bracket import make_batch
from eval_v3_shared_sparse_rule_edges import ablated_logits


def records_for_rules(gen,x,y,labels,per_pair):
    by_pair=defaultdict(list)
    for b,row in enumerate(labels):
        words=[gen.id2token[int(v)] for v in x[b]]+[gen.id2token[int(y[b,-1])]]
        for q,label in enumerate(row):
            if label!='skip_bigram':continue
            t=q+1
            found=nearest_skip_anchor(gen,words[:t])
            assert found is not None
            k,anchor=found
            trigger=words[q]
            target=gen.skip_bigrams[anchor,trigger]
            assert words[t]==target
            key=(anchor,trigger)
            if len(by_pair[key])<per_pair:
                by_pair[key].append((b,q,k,gen.token2id[target]))
    assert len(by_pair)==16 and min(len(v) for v in by_pair.values())==per_pair
    return by_pair


@torch.no_grad()
def paths(model,x,records,output_ids):
    """N×H×O signed local logit path from selected L1 source edges."""
    residual=model.blocks[0](model.embed(x))
    attn=model.blocks[1].attn
    assert not attn.qk_norm
    batch,length,width=residual.shape
    heads,dim=attn.n_head,attn.head_dim
    def shaped(linear):
        return linear(residual).reshape(batch,length,heads,dim)
    q1,k1,q2,k2,v=[shaped(w) for w in
                    (attn.q1,attn.k1,attn.q2,attn.k2,attn.v)]
    cos,sin=attn.rotary(q1)
    q1,k1=apply_rotary_emb(q1,cos,sin),apply_rotary_emb(k1,cos,sin)
    q2,k2=apply_rotary_emb(q2,cos,sin),apply_rotary_emb(k2,cos,sin)
    b=torch.tensor([r[0] for r in records],device=x.device)
    q=torch.tensor([r[1] for r in records],device=x.device)
    k=torch.tensor([r[2] for r in records],device=x.device)
    s1=(q1[b,q]*k1[b,k]).sum(-1)/dim
    s2=(q2[b,q]*k2[b,k]).sum(-1)/dim
    out=attn.out.weight.reshape(width,heads,dim).permute(1,2,0)
    edge=torch.einsum('nhd,hdc->nhc',v[b,k],out)*(s1*s2)[...,None]
    logits=torch.einsum('nhc,oc->nho',edge,model.lm_head.weight[output_ids])
    # The last block has no MLP/final norm, so ablating its one edge should
    # remove exactly this local logit vector, up to numerical precision.
    errors=[]
    for idx in (0,len(records)//2,len(records)-1):
        bb,qq,kk,_=records[idx]
        one=x[bb:bb+1]
        native=model(one)[0][0,qq,output_ids]
        for head in range(heads):
            changed=ablated_logits(model,one,
                torch.tensor([qq],device=x.device),
                torch.tensor([kk],device=x.device),head)[0,qq,output_ids]
            errors.append(float(((native-changed)-logits[idx,head]).abs().max()))
    assert max(errors)<1e-4,max(errors)
    return logits.cpu().numpy(),max(errors)


def load(path,device):
    payload=torch.load(path,map_location=device,weights_only=True)
    assert payload.get('generator')=='LanguageV3CrossedSkip'
    model=BilinearGPT(Config(**payload['config'])).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    if 'mask' in payload:
        parameters=dict(model.named_parameters())
        for name,mask in payload['mask'].items():
            assert torch.count_nonzero(parameters[name][~mask])==0
    return model,payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dense',type=Path,required=True)
    p.add_argument('--sparse',type=Path,required=True)
    p.add_argument('--seed',type=int,default=99301)
    p.add_argument('--streams',type=int,default=512)
    p.add_argument('--per-pair',type=int,default=32)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    device=torch.device(args.device)
    gen=LanguageV3CrossedSkip(seed=args.seed)
    x,y,labels,_=make_batch(gen,args.streams,129,device)
    grouped=records_for_rules(gen,x,y,labels,args.per_pair)
    outputs=gen.crossed_skip_outputs
    output_ids=[gen.token2id[w] for w in outputs]
    order=[(a,t) for a in gen.crossed_skip_anchors
           for t in gen.crossed_skip_triggers]
    records=[row for key in order for row in grouped[key]]
    matrices={}
    identity={}
    sparsity={}
    for name,path in [('dense',args.dense),('sparse',args.sparse)]:
        model,payload=load(path,device)
        values,identity[name]=paths(model,x,records,output_ids)
        centered=values-values.mean(axis=-1,keepdims=True)
        by_pair=centered.reshape(16,args.per_pair,4,4).mean(1)
        matrices[name]=dict(output_logits=by_pair.sum(1).tolist(),
                            by_head=by_pair.tolist())
        if 'mask' in payload:
            total=sum(mask.numel() for mask in payload['mask'].values())
            live=sum(int(mask.sum()) for mask in payload['mask'].values())
            sparsity[name]=dict(total=total,live=live,zero_fraction=1-live/total)
    planted=np.zeros((16,4))
    for i,(a,t) in enumerate(order):
        planted[i,outputs.index(gen.skip_bigrams[a,t])]=1
    labels=[f'{a} … {t} → {gen.skip_bigrams[a,t]}' for a,t in order]
    fig,axes=plt.subplots(1,3,figsize=(14,7.5),layout='constrained')
    axes[0].imshow(planted,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    vmax=max(np.max(np.abs(matrices[name]['output_logits'])) for name in matrices)
    for ax,name in zip(axes[1:],('dense','sparse')):
        image=ax.imshow(matrices[name]['output_logits'],vmin=-vmax,vmax=vmax,
                        cmap='coolwarm',aspect='auto')
        ax.set_title(name+' selected-anchor path')
    axes[0].set_title('Planted next token')
    for ax in axes:
        ax.set_xticks(range(4),outputs)
        ax.set_yticks(range(16),labels,fontsize=7)
    fig.colorbar(image,ax=axes[1:],label='Centered local output-logit path')
    fig.suptitle('Exact L1 QK₁×QK₂×OV→unembedding paths on fresh crossed-skip events')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output_dir/'crossed_skip_sparse_weight_paths.png',dpi=180)
    plt.close(fig)
    result=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                rule_order=labels,outputs=outputs,planted=planted.tolist(),
                matrices=matrices,identity_max_error=identity,mask_stats=sparsity,
                caveat='These exact local paths use pruned attention weights '
                       'and fresh model activations. They are not a data-free '
                       'canonicalization or an end-to-end causal sufficiency '
                       'test. Embeddings and unembedding remain dense.')
    (args.output_dir/'crossed_skip_sparse_weight_paths.json').write_text(
        json.dumps(result,indent=2)+'\n')
    print('identity',identity,'mask',sparsity,flush=True)


if __name__=='__main__':main()

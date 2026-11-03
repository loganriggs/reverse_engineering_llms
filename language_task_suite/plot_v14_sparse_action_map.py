"""Planted, dense, and pruned V14 action maps plus sparse edge ablations."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
from model import BilinearGPT, Config
from eval_v3_shared_sparse_rule_edges import ablated_logits
from surface_mixture_v2 import _split
from surface_mixture_v4 import _content
from surface_mixture_v14_conditional import ACTION, SurfaceMixtureV14Conditional
from train_v13_attributes import COMMON, REFERENCE


def load(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    assert saved['generator'] == 'SurfaceMixtureV14Conditional'
    model = BilinearGPT(Config(**saved['config'])).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    if 'mask' in saved:
        parameters = dict(model.named_parameters())
        for name, mask in saved['mask'].items():
            assert torch.count_nonzero(parameters[name][~mask]) == 0
    return model


def collect(gen, streams, length, per_attribute):
    selected = {attribute: [] for attribute in ACTION}
    for _ in range(streams):
        ids, labels, _ = gen.sample_sequence(length)
        words = [gen.id2token[v] for v in ids]
        for t, label in enumerate(labels):
            if label != 'conditional_action':
                continue
            _, raw = _split(words[:t])
            content = _content(raw)
            attribute = content[3]
            if len(selected[attribute]) >= per_attribute:
                continue
            key = t - len(raw) + raw.index('The') + 3
            assert words[key] == attribute
            assert words[t] == ACTION[attribute]
            selected[attribute].append((ids[:-1], t-1, key))
    assert all(len(rows) == per_attribute for rows in selected.values()), {
        key: len(rows) for key, rows in selected.items()}
    return selected


@torch.no_grad()
def predictions(model, rows, output_ids, device, batch_size):
    outputs = []
    for start in range(0, len(rows), batch_size):
        part = rows[start:start+batch_size]
        x = torch.tensor([r[0] for r in part], device=device)
        q = torch.tensor([r[1] for r in part], device=device)
        logits = model(x)[0].float()[torch.arange(len(part),device=device),q]
        outputs.append(logits.softmax(-1)[:, output_ids].cpu().numpy())
    return np.concatenate(outputs)


@torch.no_grad()
def ablations(model, rows, target_id, device, batch_size):
    effects = np.zeros((len(model.blocks),model.config.n_head),dtype=float)
    control = np.zeros_like(effects)
    for layer in range(len(model.blocks)):
        for head in range(model.config.n_head):
            drops = []; controls = []
            for start in range(0,len(rows),batch_size):
                part = rows[start:start+batch_size]
                x = torch.tensor([r[0] for r in part],device=device)
                q = torch.tensor([r[1] for r in part],device=device)
                k = torch.tensor([r[2] for r in part],device=device)
                idx = torch.arange(len(part),device=device)
                native = model(x)[0].float()[idx,q].softmax(-1)[:,target_id]
                removed = ablated_logits(model,x,q,k,head,layer)[idx,q].float()
                other = ablated_logits(model,x,q,q,head,layer)[idx,q].float()
                drops.extend((native-removed.softmax(-1)[:,target_id]).cpu().tolist())
                controls.extend((native-other.softmax(-1)[:,target_id]).cpu().tolist())
            effects[layer,head] = np.mean(drops)
            control[layer,head] = np.mean(controls)
    return effects,control


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dense',type=Path,required=True)
    p.add_argument('--sparse',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=99801)
    p.add_argument('--streams',type=int,default=1024)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--per-attribute',type=int,default=24)
    p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    torch.set_num_threads(1)
    device=torch.device(args.device)
    gen=SurfaceMixtureV14Conditional(seed=args.seed,**COMMON,**REFERENCE,
                                     conditional_continuation_rate=.35)
    selected=collect(gen,args.streams,args.length,args.per_attribute)
    actions=list(ACTION.values())
    ids=[gen.token2id[action] for action in actions]
    dense=load(args.dense,device)
    sparse=load(args.sparse,device)
    truth=np.eye(len(ACTION))
    dense_p=[];sparse_p=[];effect=[];control=[]
    for attribute,rows in selected.items():
        dense_p.append(predictions(dense,rows,ids,device,args.batch_size).mean(0))
        sparse_p.append(predictions(sparse,rows,ids,device,args.batch_size).mean(0))
        e,c=ablations(sparse,rows,gen.token2id[ACTION[attribute]],device,args.batch_size)
        effect.append(e.reshape(-1));control.append(c.reshape(-1))
    dense_p=np.asarray(dense_p);sparse_p=np.asarray(sparse_p)
    effect=np.asarray(effect);control=np.asarray(control)
    fig,axes=plt.subplots(1,4,figsize=(17,4.5),layout='constrained')
    for ax,title,mat in zip(axes[:3],('Planted action','Dense full-model probability',
                                    '60%-zero attention probability'),
                            (truth,dense_p,sparse_p)):
        im=ax.imshow(mat,vmin=0,vmax=1,cmap='viridis')
        ax.set_title(title)
        ax.set_xticks(range(len(actions)),actions,rotation=60,fontsize=8)
        ax.set_yticks(range(len(ACTION)),list(ACTION),fontsize=8)
    fig.colorbar(im,ax=axes[:3],fraction=.023,pad=.01,label='Full-vocabulary probability')
    bound=max(float(np.abs(effect).max()),float(np.abs(control).max()),1e-6)
    im=axes[3].imshow(effect,vmin=-bound,vmax=bound,cmap='coolwarm',aspect='auto')
    axes[3].set_title('Sparse model: remove attribute edge')
    axes[3].set_xticks(range(effect.shape[1]),
                       [f'L{l}H{h}' for l in range(len(sparse.blocks))
                        for h in range(sparse.config.n_head)],rotation=60,fontsize=8)
    axes[3].set_yticks(range(len(ACTION)),list(ACTION),fontsize=8)
    fig.colorbar(im,ax=axes[3],fraction=.07,pad=.02,label='Correct-action probability drop')
    fig.suptitle('Fresh V14 conditional-action sites; full-model edge ablation after the selected attention block')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    figure=args.output_dir/'v14_sparse_action_map_and_edges.png'
    fig.savefig(figure,dpi=180)
    plt.close(fig)
    out=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
             attributes=list(ACTION),actions=actions,planted=truth.tolist(),
             dense_full_probability=dense_p.tolist(),
             sparse_full_probability=sparse_p.tolist(),
             sparse_attribute_edge_target_probability_drop=effect.tolist(),
             sparse_query_self_edge_control_drop=control.tolist(),
             caveat='Only attention projections are pruned. Ablations remove one '
                    'specified attribute-to-query edge in one head and recompute '
                    'the entire model. A positive drop supports edge necessity '
                    'on sampled contexts, not circuit uniqueness or sufficiency.')
    (args.output_dir/'v14_sparse_action_map_and_edges.json').write_text(
        json.dumps(out,indent=2)+'\n')
    print('saved',figure,'mean attribute drop',effect.mean(),
          'mean self-edge control drop',control.mean(),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Test scalar decoders on finite incoming-head perturbations, not natural fits."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_qkov_causal_edges import load_full_model, attention_details
from run_bilin18_residual_input_forms import contract, quadratic
from run_bilin18_residual_input_forms_finite import decode
from redteam_bilin18_residual_input_linear import predict

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_residual_input_head_ablation'


@torch.inference_mode()
def main():
    torch.set_num_threads(6)
    config=json.loads((ROOT/'configs/bilin18_residual_input_head_ablation.json').read_text())
    source=np.load(ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_residual_input_forms.npz')
    decoders=np.load(ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_residual_input_forms_finite.npz')
    affines=np.load(ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_residual_input_linear_redteam.npz')
    corpus=np.load(ROOT/config['corpus'],mmap_mode='r')
    model=load_full_model('cuda');model.embedding.float();model.blocks.float()
    prepared={}
    for layer in config['layers']:
        block=model.blocks[layer]
        axes=torch.from_numpy(source[f'directions_L{layer}']).cuda().double()
        left,right,down=[m.weight.double() for m in (block.mlp.Left,block.mlp.Right,block.mlp.Down)]
        row={'axes':axes,'forms':[contract(left,right,down,u) for u in axes]}
        row['affine']={k:torch.from_numpy(affines[f'L{layer}_{k}']).cuda() for k in ('mean','scale','ym','coef')}
        for rank in (16,64):
            row[f'q{rank}']=[{k:torch.from_numpy(decoders[f'L{layer}_A{axis}_R{rank}_{k}']).cuda() for k in ('mean','linear','intercept','reads','values')} for axis in range(4)]
        prepared[layer]=row
    dim=model.config.n_embd;heads=model.config.n_head;hd=dim//heads
    methods=['exact','affine','q16','q64']
    sums={l:{m:{'error':np.zeros(heads),'target':np.zeros(heads)} for m in methods} for l in config['layers']}
    head_error=0.; head_energy=0.
    for offset in range(0,config['rows'],config['batch_size']):
        ids=torch.as_tensor(np.array(corpus[config['start']+offset:config['start']+offset+config['batch_size'],:128],dtype=np.int64),device='cuda')
        x0=F.rms_norm(model.embedding(ids),(dim,));x=x0;bus=None
        for layer,block in enumerate(model.blocks):
            mixed=block.lambdas[0]*x+block.lambdas[1]*x0
            details=attention_details(block.attn,F.rms_norm(mixed,(dim,)),bus,model.config,model.tt);bus=details['value_bus']
            pre=mixed+details['write'];normalized=F.rms_norm(pre,(dim,))
            write=block.mlp.Down(block.mlp.Left(normalized)*block.mlp.Right(normalized))
            if layer in prepared:
                row=prepared[layer];positions=config['positions']
                selected=normalized[:,positions].flatten(0,1).double()
                native=write[:,positions].flatten(0,1).double() @ row['axes'].T
                base={'exact':torch.stack([quadratic(selected,q) for q in row['forms']],-1),'affine':predict(selected,row['affine'])}
                for method in ('q16','q64'):base[method]=torch.stack([decode(selected,fit) for fit in row[method]],-1)
                attended=torch.einsum('bhpk,bkhd->bphd',details['pattern'][:,:,positions],details['mixed_value'])
                head_writes=torch.einsum('bphd,ohd->bpho',attended,block.attn.c_proj.weight.reshape(dim,heads,hd))
                native_attn=details['write'][:,positions]
                head_error+=float((head_writes.sum(2)-native_attn).square().sum());head_energy+=float(native_attn.square().sum())
                for head in range(heads):
                    changed=F.rms_norm(pre[:,positions]-head_writes[:,:,head],(dim,)).flatten(0,1)
                    changed_native=block.mlp.Down(block.mlp.Left(changed)*block.mlp.Right(changed)).double() @ row['axes'].T
                    delta=changed_native-native;changed=changed.double()
                    approximations={'exact':torch.stack([quadratic(changed,q) for q in row['forms']],-1),'affine':predict(changed,row['affine'])}
                    for method in ('q16','q64'):approximations[method]=torch.stack([decode(changed,fit) for fit in row[method]],-1)
                    for method in methods:
                        sums[layer][method]['error'][head]+=float((approximations[method]-base[method]-delta).square().sum())
                        sums[layer][method]['target'][head]+=float(delta.square().sum())
            x=pre+write+block.mlp.Down_bias
        if (offset+config['batch_size'])%16==0:print(f"head ablation rows {offset+config['batch_size']}",flush=True)
    results={str(l):{m:{'pooled_fidelity':float(1-v['error'].sum()/max(v['target'].sum(),1e-30)),'head_fidelity':(1-v['error']/np.maximum(v['target'],1e-30)).tolist(),'head_effect_energy':v['target'].tolist()} for m,v in rows.items()} for l,rows in sums.items()}
    replay=float(np.sqrt(head_error/head_energy))
    gates={'exact':replay<=1e-5 and all(r['exact']['pooled_fidelity']>=1-1e-8 for r in results.values()),'intervention_transfer':sum(r['q64']['pooled_fidelity']>=.8 for r in results.values())>=3,'beats_affine':sum(results[str(l)]['q64']['pooled_fidelity']-results[str(l)]['affine']['pooled_fidelity']>=.1 for l in (1,8,12))>=2}
    STEM.with_suffix('.json').write_text(json.dumps({'config':config,'gates':gates,'head_replay_relative_rms':replay,'results':results},indent=2,allow_nan=False))
    lines=['# Input-decoder transfer under incoming-head removal','',config['scope'],'',f'Locked gates: `{gates}`. Head-write summation relative RMS error: `{replay:.3g}`.','','| Layer | Affine finite-change fidelity | Quadratic16 | Quadratic64 |','|---|---:|---:|---:|']
    for l,r in results.items():lines.append(f"| {l} | {r['affine']['pooled_fidelity']:.3f} | {r['q16']['pooled_fidelity']:.3f} | {r['q64']['pooled_fidelity']:.3f} |")
    lines+=['','These fidelities predict intervention-induced changes, with no mean coefficient contributing to the denominator. All heads are evaluated, with natural effect-energy weighting. This neither identifies a unique semantic basis nor explains the upstream production of head values and routing.']
    STEM.with_name('BILIN18_RESIDUAL_INPUT_HEAD_ABLATION.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(gates),flush=True)


if __name__=='__main__':main()

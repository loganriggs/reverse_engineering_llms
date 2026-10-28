#!/usr/bin/env python3
"""Separate low-dimensional responses from constant predictions; plot depth results."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources
from run_bilin18_l8_context_role_compression import B
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition'
@torch.inference_mode()
def main():
    torch.set_num_threads(6);p=json.loads((ART/'bilin18_depth_source_geometry.json').read_text());cfg=p['config'];frozen=np.load(ART/'bilin18_depth_source_geometry_frozen.npz');model=load_full_model('cuda');model.embedding.float();model.blocks.float();constants={};arrays={}
    for layer in cfg['layers']:
        block=model.blocks[layer];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];panels={}
        for split,path in [('discovery',cfg['discovery']),('test',cfg['corpus'])]:
            pre,hw,_=collect_sources(model,np.load(ROOT/path,mmap_mode='r'),{**cfg,'layer':layer});x=F.rms_norm(pre,(1152,));fx=B(x,x,L,R,D);delta=[]
            for h in range(9):
                y=F.rms_norm(pre-hw[:,h],(1152,));delta.append(B(y,y,L,R,D)-fx)
            panels[split]=torch.stack(delta,1)
        mean=panels['discovery'].mean((0,1));headmean=panels['discovery'].mean(0);target=panels['test'];energy=target.square().sum(-1)
        constants[str(layer)]={}
        for name,pred in [('constant',mean),('source_constant',headmean)]:
            error=(target-pred).square().sum(-1);score=1-error.sum()/energy.sum();constants[str(layer)][name+'_fidelity']=float(score)
            arrays[f'L{layer}_{name}_error']=error.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
        arrays[f'L{layer}_energy']=energy.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
        basis=torch.from_numpy(frozen[f'L{layer}_global']).cuda();projected=(target@basis[:16].T)@basis[:16]
        constants[str(layer)]['global16_recomputed_coverage']=float(projected.square().sum()/energy.sum())
    np.savez_compressed(ART/'bilin18_depth_source_constant_rows.npz',**arrays)
    old=json.loads((ART/'bilin18_l8_document_replication.json').read_text());agreement=max(abs(p['results'][f'L8_{base}_actual_{g}_matched_r{rank}'][m]-old['results'][f'native_{base}_actual_{"tangent" if g=="tangent" else ("remove" if base=="intact" else "add")}'][str(rank)][m]) for base in ['intact','removed'] for g in ['tangent','finite'] for rank in [16,64] for m in ['pooled','site_mean','head_macro'])
    (ART/'bilin18_depth_source_constant_audit.json').write_text(json.dumps({'constant_predictions':constants,'L8_H243_actual_metric_max_difference':agreement,'scope':'Constants fit on discovery actual removals, evaluated on test actual removals. A mean-direction projection is not a constant prediction; it has a variable signed coefficient. Source-specific constants know head identity. Fidelity is1 minus squared error divided by squared target energy, not centered R2.'},indent=2))
    fig,axs=plt.subplots(1,3,figsize=(15,4.8));layers=cfg['layers'];x=np.arange(3)
    for ax,base in zip(axs[:2],['intact','removed']):
        for chart,color,label in [('global','#28649c','Shared discovery basis'),('matched','#56836a','Matched context'),('wrong_context','#b87529','Other context')]:
            vals=[p['results'][f'L{l}_{base}_actual_finite_{chart}_r16']['pooled'] for l in layers];ax.plot(x,vals,'o-',label=label,color=color)
        ax.set_xticks(x,[f'L{l}' for l in layers]);ax.set_ylim(0,1.02);ax.set_ylabel('Rank-16 actual response coverage');ax.set_title('Intact: source removal' if base=='intact' else 'Source removed: addition');ax.grid(alpha=.2)
    ax=axs[2]
    for name,color,label in [('constant','#b87529','Global constant'),('source_constant','#8b5999','Source-specific constant')]:ax.plot(x,[constants[str(l)][name+'_fidelity'] for l in layers],'o-',label=label,color=color)
    ax.plot(x,[p['results'][f'L{l}_intact_actual_finite_mean_r1']['pooled'] for l in layers],'o-',label='Mean direction (variable coefficient)',color='#28649c');ax.set_xticks(x,[f'L{l}' for l in layers]);ax.set_title('Constant output versus one direction');ax.set_ylabel('Response fidelity / projection coverage');ax.grid(alpha=.2);ax.legend(fontsize=8)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.34,.035),ncol=3,fontsize=9)
    fig.suptitle('Shared versus context-specific source-response directions across depth')
    fig.text(.5,.007,'64 discovery + 64 disjoint test documents; all 9 source heads / 1,152 outputs. Native functions, fixed16-dimensional comparisons.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.14,1,.94])
    for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_DEPTH_SOURCE_GEOMETRY.{ext}',dpi=170)
    print(json.dumps({'constant_predictions':constants,'L8_replay':agreement},indent=2))
if __name__=='__main__':main()

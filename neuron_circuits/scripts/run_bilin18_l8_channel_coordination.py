#!/usr/bin/env python3
"""Learned local channel coordination versus generic bilinear response geometry."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources
from run_bilin18_l8_context_response_geometry import source_jacobian,project
from run_bilin18_l8_context_role_compression import B,response_rules

ROOT=Path(__file__).resolve().parents[1];STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_channel_coordination'

def permuted_weights(L,R,D,p,kind):
    if kind=='right':return L,R[p],D
    if kind=='down':return L,R,D[:,p]
    if kind=='joint':return L[p],R[p],D[:,p]
    if kind=='direction':
        norms=D.norm(dim=0).clamp_min(1e-30)
        return L,R,D[:,p]/norms[p]*norms
    return L,R,D

def canonical_weights(L,R,D):
    """Remove per-factor scale/sign and the unordered input-pair ambiguity."""
    ln=L.norm(dim=1).clamp_min(1e-30);rn=R.norm(dim=1).clamp_min(1e-30)
    ls=L.gather(1,L.abs().argmax(1)[:,None]).squeeze(1).sign();rs=R.gather(1,R.abs().argmax(1)[:,None]).squeeze(1).sign()
    ls=torch.where(ls==0,torch.ones_like(ls),ls);rs=torch.where(rs==0,torch.ones_like(rs),rs)
    lc=L/(ln*ls)[:,None];rc=R/(rn*rs)[:,None];dc=D*(ln*ls*rn*rs)
    first=(lc!=rc).to(torch.int64).argmax(1)
    swap=lc.gather(1,first[:,None]).squeeze(1)>rc.gather(1,first[:,None]).squeeze(1)
    return torch.where(swap[:,None],rc,lc),torch.where(swap[:,None],lc,rc),dc

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());torch.manual_seed(cfg['seed'])
    model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[8]
    native_weights=tuple(m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]);width=native_weights[0].shape[0]
    original_weights=native_weights
    if cfg.get('canonicalize'):native_weights=canonical_weights(*native_weights)
    permutations={r:torch.randperm(width,device='cuda',generator=torch.Generator(device='cuda').manual_seed(cfg['seed']+r)) for r in cfg['permutation_replicates']}
    cases={'native':('native',0),'joint':('joint',0),**{f'{kind}{r}':(kind,r) for kind in ['right','down'] for r in cfg['permutation_replicates']}}
    if cfg.get('direction_controls'):cases.update({f'direction{r}':('direction',r) for r in cfg['permutation_replicates']})
    cache=np.load(STEM.parent/'bilin18_l8_coverage_grid_train_validation.npz')
    train={k:torch.from_numpy(cache['train_'+k]).cuda() for k in ['native']+[f'drop{h}' for h in range(9)]}
    global_bases={};arrays={f'permutation{r}':p.cpu().numpy() for r,p in permutations.items()}
    for name,(kind,rep) in cases.items():
        L,R,D=permuted_weights(*native_weights,permutations[rep],kind)
        native=B(train['native'],train['native'],L,R,D);gram=torch.zeros(1152,1152,dtype=torch.double,device='cuda')
        for h in range(9):
            y=train[f'drop{h}'];delta=(B(y,y,L,R,D)-native).double();gram+=delta.T@delta
        _,vectors=torch.linalg.eigh(gram.cpu());basis=vectors[:,-64:].flip(1).T.float().cuda();global_bases[name]=basis;arrays[name+'_global_basis']=basis.cpu().numpy()
        print('frozen global basis',name,flush=True)
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**arrays)
    print('All permutations/bases FROZEN before test',flush=True)
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,hw,replay=collect_sources(model,corpus,cfg);x=F.rms_norm(pre,(1152,));N=len(x)
    source_bases=[torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
    context=np.load(STEM.parent/'bilin18_l8_context_role_compression_frozen.npz');mean=torch.from_numpy(context['mean']).cuda()
    codes={r:(torch.from_numpy(context[f'pca{r}_encoder']).cuda(),torch.from_numpy(context[f'pca{r}_decoder']).cuda()) for r in cfg['context_ranks']}
    wrong=(torch.arange(N,device='cuda')+len(cfg['positions']))%N;rows={};results={};head_results={};isotropic={};joint_error=0.;native_energy=0.;orth_error=0.
    native_function=B(x,x,*native_weights)
    canonical_replay=float((native_function-B(x,x,*original_weights)).norm()/native_function.norm())
    for name,(kind,rep) in cases.items():
        L,R,D=permuted_weights(*native_weights,permutations[rep],kind);fx=B(x,x,L,R,D)
        if name=='joint':joint_error=float((fx-native_function).square().sum());native_energy=float(native_function.square().sum())
        isotropic[name]={}
        for h in range(9):
            charts=[];tangents=[];squares=[];source=source_bases[h]
            for offset in range(0,N,cfg['matrix_batch']):
                J=source_jacobian(pre[offset:offset+cfg['matrix_batch']],L,R,D,source);u,s,_=torch.linalg.svd(J,full_matrices=False);u=torch.linalg.qr(u[:,:,:64]).Q
                charts.append(u.transpose(1,2));squares.append(s.square());tangents.append(-torch.einsum('ndr,nr->nd',J,hw[offset:offset+len(J),h]@source))
                orth_error=max(orth_error,float((u.transpose(1,2)@u-torch.eye(64,device='cuda')).abs().max()))
            chart=torch.cat(charts);tangent=torch.cat(tangents);sq=torch.cat(squares);isotropic[name][str(h)]={str(r):float(sq[:,:r].sum()/sq.sum()) for r in cfg['ranks']}
            for scale in cfg['scales']:
                y=F.rms_norm(pre-scale*hw[:,h],(1152,));target=B(y,y,L,R,D)-fx;energy=target.square().sum(-1);prefix=f'{name}_s{scale:g}'
                rows.setdefault(prefix+'_energy',np.zeros(cfg['rows']));rows[prefix+'_energy']+=energy.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                predictions={'full_tangent':scale*tangent};captures={}
                for rank in cfg['ranks']:
                    for family,basis in [('matched',chart[:,:rank]),('wrong_context',chart[wrong,:rank]),('global',global_bases[name][:rank])]:
                        key=f'{family}{rank}';captures[key]=project(target,basis).square().sum(-1);predictions[key]=project(scale*tangent,basis)
                for rank,(E,A) in codes.items():
                    rules=response_rules(x,y,mean,E,A,L,R,D)
                    for rule in ['context_only','cross']:predictions[f'pca{rank}_{rule}']=rules[rule]
                head_results[f'{name}_h{h}_s{scale:g}']={}
                for key,prediction in predictions.items():
                    error=(prediction-target).square().sum(-1);rowkey=prefix+'_'+key
                    rows.setdefault(rowkey+'_error',np.zeros(cfg['rows']));rows[rowkey+'_error']+=error.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                    metric={'prediction_fidelity':float(1-error.sum()/energy.sum())}
                    if key in captures:
                        rows.setdefault(rowkey+'_captured',np.zeros(cfg['rows']));rows[rowkey+'_captured']+=captures[key].reshape(cfg['rows'],-1).sum(1).cpu().numpy();metric['coverage']=float(captures[key].sum()/energy.sum())
                    head_results[f'{name}_h{h}_s{scale:g}'][key]=metric
            print(name,'head',h,'complete',flush=True)
        for scale in cfg['scales']:
            prefix=f'{name}_s{scale:g}';energy=rows[prefix+'_energy'];results[prefix]={}
            for key in predictions:
                metric={'prediction_fidelity':float(1-rows[prefix+'_'+key+'_error'].sum()/energy.sum())}
                if prefix+'_'+key+'_captured' in rows:metric['coverage']=float(rows[prefix+'_'+key+'_captured'].sum()/energy.sum())
                results[prefix][key]=metric
        STEM.with_name(STEM.name+'_partial.json').write_text(json.dumps({'config':cfg,'results':results},indent=2))
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows)
    rng=np.random.default_rng(cfg['seed']);ix=rng.integers(cfg['rows'],size=(2000,cfg['rows']));contrasts={}
    for name in cases:
        if name in ['native','joint']:continue
        contrasts[name]={}
        for metric in ['matched16','global16','pca64_cross']:
            suffix='_captured' if metric in ['matched16','global16'] else '_error'
            ns=rows['native_s1_'+metric+suffix];cs=rows[name+'_s1_'+metric+suffix];ne=rows['native_s1_energy'];ce=rows[name+'_s1_energy']
            nd=ns.sum()/ne.sum();cd=cs.sum()/ce.sum();boot=ns[ix].sum(1)/ne[ix].sum(1)-cs[ix].sum(1)/ce[ix].sum(1)
            if suffix=='_error':nd,cd,boot=1-nd,1-cd,-boot
            contrasts[name][metric]={'native_minus_control':float(nd-cd),'paired_row_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
    joint_metric_error=max(abs(results['native_s'+strscale][m][k]-results['joint_s'+strscale][m][k]) for strscale in ['0.05','1'] for m,r in results['native_s'+strscale].items() for k in r)
    get=lambda name:results[name+'_s1']
    shuffles=[n for n in cases if n not in ['native','joint']]
    gates={'invariant_control':(joint_error/native_energy)**.5<=1e-4 and joint_metric_error<=1e-3,
           'right_pairing_specificity':all(get('native')['matched16']['coverage']-get(f'right{r}')['matched16']['coverage']>=.1 for r in cfg['permutation_replicates']),
           'down_binding_specificity':all(get('native')['matched16']['coverage']-get(f'down{r}')['matched16']['coverage']>=.1 for r in cfg['permutation_replicates']),
           'context_rotation_survives':all(get(n)['matched16']['coverage']-get(n)['wrong_context16']['coverage']>=.1 for n in shuffles),
           'broad_context_survives':all(get(n)['pca64_cross']['prediction_fidelity']<.8 for n in shuffles)}
    output={'config':cfg,'gates':gates,'head_replay':replay,'canonical_native_write_relative_rms':canonical_replay,'orthogonality_error':orth_error,'joint_native_write_relative_rms':(joint_error/native_energy)**.5,'joint_metric_max_difference':joint_metric_error,'results':results,'head_results':head_results,'isotropic_spectral_capture':isotropic,'contrasts':contrasts}
    STEM.with_suffix('.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    lines=['# Learned local coordination versus bilinear geometry','','Every score targets its own surrogate function on identical trained upstream states/head writes. Global bases are refit separately.','','| Local MLP | Matched16 coverage | Wrong-context16 | Global16 | Matched64 | Full tangent | PCA64 cross prediction | PCA256 cross prediction |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in cases:
        r=get(name);vals=[r['matched16']['coverage'],r['wrong_context16']['coverage'],r['global16']['coverage'],r['matched64']['coverage'],r['full_tangent']['prediction_fidelity'],r['pca64_cross']['prediction_fidelity'],r['pca256_cross']['prediction_fidelity']]
        lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    lines+=['',f'Locked gates: {gates}. Joint permutation write replay relative RMS {(joint_error/native_energy)**.5:.3g}; largest pooled metric change {joint_metric_error:.3g}.','',cfg['scope']]
    STEM.with_name(STEM.name.upper()+'.md').write_text('\n'.join(lines)+'\n');print(gates,flush=True)

if __name__=='__main__':
    import sys
    if '--canonical' in sys.argv:STEM=STEM.with_name('bilin18_l8_canonical_channel_coordination')
    main()

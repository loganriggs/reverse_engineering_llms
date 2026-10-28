"""H272 directly optimize fixed-rank partner projection, frozen native coefficients."""
import json,hashlib
import sys
from types import SimpleNamespace
def minimum_norm(gram):
    w=np.ones(len(gram))/len(gram)
    for iteration in range(20000):
        gw=gram@w;j=int(np.argmin(gw));gap=float(w@gw-gw[j])
        if gap<1e-10:break
        d=-w.copy();d[j]+=1;alpha=min(1.,max(0.,gap/float(d@gram@d)));w+=alpha*d
    return SimpleNamespace(x=w,success=gap<1e-8,message=f"Frank-Wolfe iteration {iteration}, gap {gap}")
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_partner_compression import cross_response
from run_bilin18_l17_core_interaction_factorial import quadratic_terms
from run_bilin18_l17_residual_term_accounting import mean_linear
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_direct_partner_fit'
def partner_q(w,core):return torch.linalg.qr(w-core@(core.T@w),mode='reduced').Q

@torch.inference_mode()
def main():
    global STEM
    if len(sys.argv)>1:STEM=ART/sys.argv[1]
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float()
    L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');f=load_program(ART/'bilin18_l17_conditional_partner_rank_bases.npz','cuda');c=torch.linalg.qr(f['core']).Q;mu=f['mean'];base={n:partner_q(f[n][:,:cfg.get('rank',32)],c) for n in ['weighted','random0']}
    with torch.no_grad():
        pre,heads,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':1024});x=F.rms_norm(pre,(1152,));zs_actual=[F.rms_norm(pre-heads[:,h],(1152,)) for h in cfg['heads']];zs_angle=[];rng=torch.Generator(device='cuda').manual_seed(cfg['payload_seed'])
        for h in cfg['heads']:
            source=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;pay=angle_matched_payload(pre,heads[:,h],source,torch.randn(len(pre),128,generator=rng,device='cuda'));zs_angle.append(F.rms_norm(pre-pay,(1152,)))
    path=ART/'bilin18_l17_document_repartition_responses.npy';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['response_sha256'];Y=torch.tensor(np.load(path),device='cuda').reshape(18,4096,1152).double();path=ART/'bilin18_l17_shared_kernel_mixed_frozen.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['mixed_sha256'];f=np.load(path);ref=[torch.tensor(f[k],device='cuda') for k in ['ux','vx','uz','vz']];assignment=torch.tensor(f['assignment'],device='cuda');labels=Y[assignment,torch.arange(3072,device='cuda')];scale=float(f['kernel_scale']);base=load_program(ART/'bilin18_l17_partner_common_descent_rank64_frozen.npz','cuda');q=partner_q(base['optimized'],c).double();cc=c.double()
    def coords(v):
        w=v.double()-mu.double();u=w@cc;return u,w-u@cc.T-(w@q)@q.T
    def kernel(a,b):
        u,v,w,t=a;uu,vv,ww,tt=b;return ((w@ww.T)*(t@tt.T)+(u@uu.T)*(v@vv.T)-(w@uu.T)*(t@vv.T)-(u@ww.T)*(v@tt.T))/scale
    K=kernel(ref,ref);ev,U=torch.linalg.eigh((K+K.T)/2);minimum=float(ev.min());ev=ev.clamp_min(0);rhs=U.T@labels;coef=U@(rhs/(ev+.01)[:,None]);replay=float((coef-torch.tensor(f['coef'],device='cuda')).norm()/coef.norm());assert replay<1e-6;u,v=coords(x[3072:]);rows={}
    for g,z in enumerate(zs_actual+zs_angle):
        w,t=coords(z[3072:]);left=kernel((u,v,w,t),ref)@U;target=Y[g,3072:];rows[f'g{g}_energy']=target.square().sum(-1).reshape(256,4).sum(1).cpu().numpy()
        for alpha in cfg['ridge']:
            pred=(left/(ev+alpha)[None,:])@rhs;rows[f'g{g}_a{alpha}_error']=(pred-target).square().sum(-1).reshape(256,4).sum(1).cpu().numpy()
    results={};energy={};errors={}
    for family,groups in [('actual',range(9)),('angle',range(9,18))]:
        energy[family]=sum(rows[f'g{g}_energy'] for g in groups);results[family]={};errors[family]={}
        for alpha in cfg['ridge']:
            err=sum(rows[f'g{g}_a{alpha}_error'] for g in groups);errors[family][str(alpha)]=err;results[family][str(alpha)]={'calibration':float(1-err[:128].sum()/energy[family][:128].sum()),'evaluation':float(1-err[128:].sum()/energy[family][128:].sum())}
    selected=max([str(v) for v in cfg['ridge']],key=lambda k:min(results[f][k]['calibration'] for f in results));boot=np.random.default_rng(20270216).integers(128,size=(2000,128));contrasts={}
    for family in results:
        gain=(errors[family]['0.01']-errors[family][selected])[128:];e=energy[family][128:];contrasts[family]={'gain':float(gain.sum()/e.sum()),'paired95':np.quantile(gain[boot].sum(1)/e[boot].sum(1),[.025,.975]).tolist()}
    gates={'shared10':all(v[selected]['evaluation']>=.1 for v in results.values()),'gain5':all(v['gain']>=.05 for v in contrasts.values())};coef=U@(rhs/(ev+float(selected))[:,None]);np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**{k:f[k] for k in ['ux','vx','uz','vz','assignment','kernel_scale']},coef=coef.cpu().numpy(),ridge=np.array(float(selected)));np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'selected':selected,'contrasts':contrasts,'gates':gates,'coefficient_replay_relative':replay,'min_eigenvalue':minimum},indent=2)+'\n');print('FINISHED',results,'selected',selected,contrasts,gates,flush=True)
if __name__=='__main__':main()

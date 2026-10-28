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
    assert hashlib.sha256((ART/'bilin18_l17_interaction_kernel.json').read_bytes()).hexdigest()==cfg['selection_sha256'];path=ART/'bilin18_l17_document_repartition_responses.npy';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['response_sha256'];Y=torch.tensor(np.load(path),device='cuda').reshape(18,4096,1152)[:,:3072].double();xr=x[:3072];zr=[z[:3072] for z in zs_actual+zs_angle]
    pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),{**cfg,'rows':128});x=F.rms_norm(pre,(1152,));gen=torch.Generator(device='cuda').manual_seed(cfg['fresh_payload_seed']);zs=[]
    for h in range(9):
        source=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;pay=angle_matched_payload(pre,heads[:,h],source,torch.randn(len(pre),128,generator=gen,device='cuda'));zs.append(F.rms_norm(pre-pay,(1152,)))
    zs=[F.rms_norm(pre-heads[:,h],(1152,)) for h in range(9)]+zs;frozen=load_program(ART/'bilin18_l17_partner_common_descent_rank64_frozen.npz','cuda');q=partner_q(frozen['optimized'],c);cc=c.double();qq=q.double();rows={};audits=[];directory=STEM.with_name(STEM.name+'_frozen');directory.mkdir(exist_ok=True)
    def coords(v):
        w=v.double()-mu.double();u=w@cc;return u,w-u@cc.T-(w@qq)@qq.T
    def interaction(a,b):
        u,v,w,t=a;uu,vv,ww,tt=b;return (w@ww.T)*(t@tt.T)+(u@uu.T)*(v@vv.T)-(w@uu.T)*(t@vv.T)-(u@ww.T)*(v@tt.T)
    ux,vx=coords(xr);ut,vt=coords(x);np.savez_compressed(directory/'common.npz',core=c.cpu().numpy(),partner=q.cpu().numpy(),mean=mu.cpu().numpy(),ux=ux.cpu().numpy(),vx=vx.cpu().numpy())
    for g,z in enumerate(zs):
        uz,vz=coords(zr[g]);uzt,vzt=coords(z);tr=(ux,vx,uz,vz);te=(ut,vt,uzt,vzt);K=interaction(tr,tr);Kt=interaction(te,tr);blocks=[(ux+uz)/2,(vx+vz)/2,uz-ux,vz-vx];testblocks=[(ut+uzt)/2,(vt+vzt)/2,uzt-ut,vzt-vt];add=torch.zeros_like(K);addt=torch.zeros_like(Kt);scales=[]
        for b,t in zip(blocks,testblocks):
            scale=b.square().sum(1).mean();scales.append(float(scale));add+=b@b.T/scale;addt+=t@b.T/scale
        target=torch.cat([quadratic_terms(x[i:i+256],z[i:i+256],mu,c,L,R,D,p)[:,1]-cross_response(x[i:i+256],z[i:i+256],mu,c,q,L,R,D,p) for i in range(0,512,256)]).double();rows[f'g{g}_energy']=target.square().sum(-1).reshape(128,4).sum(1).cpu().numpy();saved={'uz':uz.cpu().numpy(),'vz':vz.cpu().numpy(),'add_block_scales':np.array(scales)}
        for name,kk,kt in [('interaction',K,Kt),('additive',add,addt)]:
            scale=kk.diag().mean();kk=kk/scale;kt=kt/scale;alpha=float(cfg['selected']['actual' if g<9 else 'angle'][name]);system=(kk+kk.T)/2+alpha*torch.eye(3072,device='cuda',dtype=torch.float64);factor=torch.linalg.cholesky(system);coef=torch.cholesky_solve(Y[g],factor);residual=float((system@coef-Y[g]).norm()/Y[g].norm());assert residual<1e-8;pred=kt@coef;rows[f'g{g}_{name}_error']=(pred-target).square().sum(-1).reshape(128,4).sum(1).cpu().numpy();saved[name+'_coef']=coef.cpu().numpy();saved[name+'_kernel_scale']=np.array(float(scale));audits.append({'group':g,'kernel':name,'solve_relative':residual})
        np.savez_compressed(directory/f'g{g}.npz',**saved);print('group',g,'complete',flush=True)
    boot=np.random.default_rng(20270212).integers(128,size=(2000,128));summary={};contrasts={}
    for family,groups in [('actual',range(9)),('angle',range(9,18))]:
        energy=sum(rows[f'g{g}_energy'] for g in groups);summary[family]={};errors={}
        for name in ['interaction','additive']:
            err=sum(rows[f'g{g}_{name}_error'] for g in groups);errors[name]=err;summary[family][name]={'fidelity':float(1-err.sum()/energy.sum()),'paired95':np.quantile(1-err[boot].sum(1)/energy[boot].sum(1),[.025,.975]).tolist()}
        gain=errors['additive']-errors['interaction'];contrasts[family]={'gain':float(gain.sum()/energy.sum()),'paired95':np.quantile(gain[boot].sum(1)/energy[boot].sum(1),[.025,.975]).tolist()}
    gates={'interaction_gain':all(v['gain']>=.05 and v['paired95'][0]>0 for v in contrasts.values()),'interaction20':all(v['interaction']['fidelity']>=.2 for v in summary.values())};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'contrasts':contrasts,'gates':gates,'solve_audits':audits},indent=2)+'\n');print('FINISHED',summary,contrasts,gates,flush=True)
if __name__=='__main__':main()

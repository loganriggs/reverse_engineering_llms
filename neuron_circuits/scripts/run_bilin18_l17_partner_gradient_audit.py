"""H272 directly optimize fixed-rank partner projection, frozen native coefficients."""
import json
import sys
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

def main():
    global STEM
    if len(sys.argv)>1:STEM=ART/sys.argv[1]
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float()
    L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');f=load_program(ART/'bilin18_l17_conditional_partner_rank_bases.npz','cuda');c=torch.linalg.qr(f['core']).Q;mu=f['mean'];base={n:partner_q(f[n][:,:cfg.get('rank',32)],c) for n in ['weighted','random0']}
    with torch.no_grad():
        pre,heads,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':1024});x=F.rms_norm(pre,(1152,));zs_actual=[F.rms_norm(pre-heads[:,h],(1152,)) for h in cfg['heads']];zs_angle=[];rng=torch.Generator(device='cuda').manual_seed(cfg['payload_seed'])
        for h in cfg['heads']:
            source=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;pay=angle_matched_payload(pre,heads[:,h],source,torch.randn(len(pre),128,generator=rng,device='cuda'));zs_angle.append(F.rms_norm(pre-pay,(1152,)))
        zs=zs_actual+zs_angle;cut=768*4;tx=torch.cat([x[:cut] for _ in zs]);tz=torch.cat([z[:cut] for z in zs]);vx=torch.cat([x[cut:] for _ in zs]);vz=torch.cat([z[cut:] for z in zs])
        def targets(xx,zz):return torch.cat([quadratic_terms(xx[i:i+256],zz[i:i+256],mu,c,L,R,D,p)[:,1] for i in range(0,len(xx),256)])
        ty=targets(tx,tz);vy=targets(vx,vz);normalizer=ty.square().sum(1).mean();print('targets cached',flush=True)
    @torch.no_grad()
    def score(q,xx,zz,yy):
        errors=0.
        for i in range(0,len(xx),256):errors+=(cross_response(xx[i:i+256],zz[i:i+256],mu,c,q,L,R,D,p)-yy[i:i+256]).square().sum().double()
        return float(1-errors/yy.square().sum().double())
    q0=base['weighted'];half=len(tx)//2;vh=len(vx)//2;grads={}
    for label,sl in [('actual',slice(0,half)),('angle',slice(half,None))]:
        xx,zz,yy=tx[sl],tz[sl],ty[sl];w=q0.clone().requires_grad_();den=yy.square().sum().double();gradient=torch.zeros_like(w)
        for i in range(0,len(xx),256):
            pred=cross_response(xx[i:i+256],zz[i:i+256],mu,c,partner_q(w,c),L,R,D,p);loss=(pred-yy[i:i+256]).square().sum().double()/den;gradient+=torch.autograd.grad(loss,w)[0]
        gradient=gradient-c@(c.T@gradient)-q0@(q0.T@gradient);grads[label]=gradient;print('GRADIENT',label,float(gradient.norm()),flush=True)
    ga,gb=grads['actual'],grads['angle'];cos=float((ga*gb).sum()/(ga.norm()*gb.norm()));directions={'unit_sum':-(ga/ga.norm()+gb/gb.norm()),'mean_loss':-(ga+gb)};directions={n:d/d.norm() for n,d in directions.items()}
    @torch.no_grad()
    def allscores(q):
        return {'actual_train':score(q,tx[:half],tz[:half],ty[:half]),'angle_train':score(q,tx[half:],tz[half:],ty[half:]),'actual_validation':score(q,vx[:vh],vz[:vh],vy[:vh]),'angle_validation':score(q,vx[vh:],vz[vh:],vy[vh:])}
    baseline=allscores(q0);curves={};frozen={'proxy':q0};derivatives={};finite={}
    for name,d in directions.items():
        derivatives[name]={label:float((g*d).sum()) for label,g in grads.items()};curves[name]=[]
        for step in cfg['step_norms']:
            q=partner_q(q0+step*d,c);values=allscores(q);curves[name].append({'step':step,**values});frozen[name+'_s'+str(step)]=q;print(name,step,values,flush=True)
        eps=cfg['finite_difference_step'];plus=allscores(partner_q(q0+eps*d,c));minus=allscores(partner_q(q0-eps*d,c));finite[name]={label:-(plus[label+'_train']-minus[label+'_train'])/(2*eps) for label in ['actual','angle']}
    improving=[{'direction':n,**v} for n,vs in curves.items() for v in vs if all(v[label+'_train']>baseline[label+'_train'] for label in ['actual','angle'])];gates={'common_training_descent':bool(improving),'validation_both_gain':any(all(v[label+'_validation']-baseline[label+'_validation']>=.001 for label in ['actual','angle']) for vs in curves.values() for v in vs)}
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),core=c.cpu().numpy(),mean=mu.cpu().numpy(),**{n:v.detach().cpu().numpy() for n,v in frozen.items()},**{'gradient_'+n:g.cpu().numpy() for n,g in grads.items()});STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'gradient_norms':{n:float(g.norm()) for n,g in grads.items()},'gradient_cosine':cos,'loss_directional_derivatives':derivatives,'finite_difference_derivatives':finite,'baseline':baseline,'curves':curves,'gates':gates},indent=2)+'\n');print('GATES',gates,'COSINE',cos,flush=True)
if __name__=='__main__':main()

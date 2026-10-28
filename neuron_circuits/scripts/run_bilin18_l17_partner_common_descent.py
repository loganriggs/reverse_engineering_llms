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
    q=base['weighted'];half=len(tx)//2;vh=len(vx)//2
    @torch.no_grad()
    def train_score(q):return {'actual':score(q,tx[:half],tz[:half],ty[:half]),'angle':score(q,tx[half:],tz[half:],ty[half:])}
    @torch.no_grad()
    def validation_score(q):return {'actual':score(q,vx[:vh],vz[:vh],vy[:vh]),'angle':score(q,vx[vh:],vz[vh:],vy[vh:])}
    history=[{'iteration':0,'train':train_score(q),'validation':validation_score(q)}];frozen={'step0':q.clone()};stop='budget'
    for iteration in range(1,cfg['iterations']+1):
        grads={}
        for label,sl in [('actual',slice(0,half)),('angle',slice(half,None))]:
            xx,zz,yy=tx[sl],tz[sl],ty[sl];w=q.clone().requires_grad_();den=yy.square().sum().double();g=torch.zeros_like(w)
            for i in range(0,len(xx),256):
                pred=cross_response(xx[i:i+256],zz[i:i+256],mu,c,partner_q(w,c),L,R,D,p);loss=(pred-yy[i:i+256]).square().sum().double()/den;g+=torch.autograd.grad(loss,w)[0]
            grads[label]=g-c@(c.T@g)-q@(q.T@g)
        norms={n:float(g.norm()) for n,g in grads.items()}
        if min(norms.values())<1e-12:stop='small_gradient';break
        ga,gb=grads['actual'],grads['angle'];cos=float((ga*gb).sum()/(ga.norm()*gb.norm()));d=-(ga/ga.norm()+gb/gb.norm())
        if float(d.norm())<1e-10:stop='opposing_gradients';break
        d=d/d.norm();derivatives={n:float((g*d).sum()) for n,g in grads.items()};trials=[];accepted=False;old=history[-1]['train']
        for alpha in cfg['step_grid']:
            candidate=partner_q(q+alpha*d,c);values=train_score(candidate);good=all(values[n]>=old[n]-cfg['armijo']*alpha*derivatives[n] for n in grads);trials.append({'step':alpha,'train':values,'accepted':good})
            if good:q=candidate.detach();accepted=True;break
        record={'iteration':iteration,'norms':norms,'cosine':cos,'loss_derivatives':derivatives,'trials':trials,'accepted':accepted}
        if not accepted:stop='line_search_exhausted';history.append(record);break
        record.update(train=values,validation=validation_score(q),step=alpha);history.append(record);frozen['step'+str(iteration)]=q.clone();print('ITERATION',iteration,record,flush=True)
    valid=[v for v in history if 'validation' in v];selected=max(valid,key=lambda v:min(v['validation'].values()));winner='step'+str(selected['iteration']);frozen['optimized']=frozen[winner];frozen['weighted']=frozen['step0'];gates={'validation_both_gain':all(selected['validation'][n]-history[0]['validation'][n]>=.02 for n in ['actual','angle']),'compact80':min(selected['validation'].values())>=.8,'monotone_train':all(all(v['train'][n]>=u['train'][n] for n in ['actual','angle']) for u,v in zip(valid,valid[1:]))};np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),core=c.cpu().numpy(),mean=mu.cpu().numpy(),**{n:v.cpu().numpy() for n,v in frozen.items()});STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'history':history,'winner':winner,'selected':selected,'stop_reason':stop,'gates':gates},indent=2)+'\n');print('FINISHED',winner,stop,gates,flush=True)
if __name__=='__main__':main()

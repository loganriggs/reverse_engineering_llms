"""H272 directly optimize fixed-rank partner projection, frozen native coefficients."""
import json
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
    saved=load_program(ART/'bilin18_l17_partner_common_descent_rank64_frozen.npz','cuda');out={};arrays={};nt=len(tx)//18;nv=len(vx)//18
    def pieces(xx,zz,yy,size,i):return xx[i*size:(i+1)*size],zz[i*size:(i+1)*size],yy[i*size:(i+1)*size]
    @torch.no_grad()
    def scores(q,xx,zz,yy,size):return [score(q,*pieces(xx,zz,yy,size,i)) for i in range(18)]
    energies=torch.stack([ty[i*nt:(i+1)*nt].square().sum().double() for i in range(18)]);weights=energies.reshape(2,9);weights=weights/weights.sum(1,keepdim=True)
    q=partner_q(saved['optimized'],c);history=[{'iteration':0,'train':scores(q,tx,tz,ty,nt),'validation':scores(q,vx,vz,vy,nv)}];frozen={'step0':q.clone()};stop='budget'
    for iteration in range(1,cfg['iterations']+1):
        gs=[]
        for i in range(18):
            xx,zz,yy=pieces(tx,tz,ty,nt,i);w=q.clone().requires_grad_();g=torch.zeros_like(w)
            for start in range(0,nt,256):
                pred=cross_response(xx[start:start+256],zz[start:start+256],mu,c,partner_q(w,c),L,R,D,p);loss=(pred-yy[start:start+256]).square().sum().double()/energies[i];g+=torch.autograd.grad(loss,w)[0]
            gs.append(g-c@(c.T@g)-q@(q.T@g))
        g=torch.stack(gs);norms=g.flatten(1).norm(dim=1)
        if float(norms.min())<1e-12:stop='small_gradient';break
        units=g/norms[:,None,None];gram=(units.flatten(1)@units.flatten(1).T).double().cpu().numpy();opt=minimum_norm(gram);d=-(units*torch.tensor(opt.x,device='cuda',dtype=g.dtype)[:,None,None]).sum(0)
        if float(d.norm())<1e-10:stop='small_direction';break
        d=d/d.norm();deriv=(g*d).flatten(1).sum(1).cpu().numpy()
        if np.max(deriv)>=0:stop='no_verified_common_direction';break
        old=np.array(history[-1]['train']);trials=[];accepted=False
        for step in cfg['step_grid']:
            candidate=partner_q(q+step*d,c);values=np.array(scores(candidate,tx,tz,ty,nt));good=bool(np.all(values>=old-cfg['armijo']*step*deriv));trials.append({'step':step,'train':values.tolist(),'accepted':good})
            if good:q=candidate.detach();accepted=True;break
        record={'iteration':iteration,'norms':norms.cpu().tolist(),'derivatives':deriv.tolist(),'solver_success':bool(opt.success),'solver_message':opt.message,'trials':trials,'accepted':accepted}
        if not accepted:history.append(record);stop='line_search_exhausted';break
        record.update(step=step,train=values.tolist(),validation=scores(q,vx,vz,vy,nv));history.append(record);frozen['step'+str(iteration)]=q.clone();print('ITERATION',iteration,'step',step,'min validation gain',float(np.min(np.array(record['validation'])-history[0]['validation'])),flush=True)
    valid=[v for v in history if 'validation' in v];selected=max(valid,key=lambda v:np.min(np.array(v['validation'])-valid[0]['validation']));winner='step'+str(selected['iteration']);gain=np.array(selected['validation'])-valid[0]['validation'];gates={'allgroup_validation_gain':bool(gain.min()>=.001),'mean_validation_gain':bool(gain.mean()>=.01),'monotone_train':all(np.all(np.array(v['train'])>=u['train']) for u,v in zip(valid,valid[1:]))};frozen['optimized']=frozen[winner];np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),core=c.cpu().numpy(),mean=mu.cpu().numpy(),**{n:v.cpu().numpy() for n,v in frozen.items()});STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'history':history,'winner':winner,'selected':selected,'stop_reason':stop,'gates':gates},indent=2)+'\n');print('FINISHED',winner,stop,gates,flush=True)
if __name__=='__main__':main()

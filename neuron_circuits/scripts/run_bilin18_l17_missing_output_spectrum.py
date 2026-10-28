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
    q=partner_q(saved['optimized'],c);grams={};audit=[]
    for split,xx,zz,yy,size in [('train',tx,tz,ty,nt),('validation',vx,vz,vy,nv)]:
        covs=[]
        for group in range(18):
            u,v,y=pieces(xx,zz,yy,size,group);cov=torch.zeros(1152,1152,device='cuda',dtype=torch.float64);energy=0.
            for start in range(0,size,256):
                err=(y[start:start+256]-cross_response(u[start:start+256],v[start:start+256],mu,c,q,L,R,D,p)).double();cov+=err.T@err;energy+=float(err.square().sum())
            audit.append(abs(float(cov.trace())-energy)/energy);covs.append(cov)
        grams[split]=torch.stack(covs);print(split,'covariance complete',flush=True)
    train=grams['train'];traces=train.diagonal(dim1=1,dim2=2).sum(1);basis={};spectra={}
    for name,cov in [('pooled',train.sum(0)),('equal_group',(train/traces[:,None,None]).sum(0))]:
        eig,U=torch.linalg.eigh(cov);basis[name]=U.flip(1);spectra[name]=(eig.flip(0)/eig.sum()).cpu().tolist()
    gen=torch.Generator(device='cuda').manual_seed(20270129);basis['random']=torch.linalg.qr(torch.randn(1152,256,device='cuda',dtype=torch.float64,generator=gen)).Q
    results={}
    for name,U in basis.items():
        results[name]={}
        for k in cfg['output_ranks']:
            V=U[:,:k];results[name][str(k)]={}
            for split,covs in grams.items():
                total=covs.diagonal(dim1=1,dim2=2).sum(1);captured=torch.einsum('ik,gij,jk->g',V,covs,V);results[name][str(k)][split]={'groups':(captured/total).cpu().tolist(),'pooled':float(captured.sum()/total.sum()),'equal_group':float((captured/total).mean())}
    v32=np.array(results['equal_group']['32']['validation']['groups']);v128=np.array(results['equal_group']['128']['validation']['groups']);random=np.array(results['random']['32']['validation']['groups']);gates={'compact32':bool(v32.min()>=.8),'compact128':bool(v128.min()>=.8),'specificity32':bool(np.min(v32-random)>=.1)}
    np.savez_compressed(STEM.with_name(STEM.name+'_statistics.npz'),**{n:v.cpu().numpy() for n,v in grams.items()});np.savez_compressed(STEM.with_name(STEM.name+'_bases.npz'),**{n:v[:,:256].cpu().numpy() for n,v in basis.items()});STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'spectra':spectra,'trace_audit_max_relative':max(audit),'gates':gates},indent=2)+'\n');print('GATES',gates,flush=True)
if __name__=='__main__':main()

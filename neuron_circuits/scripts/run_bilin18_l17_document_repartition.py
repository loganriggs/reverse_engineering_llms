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
    q=partner_q(saved['optimized'],c);responses=[]
    for group in range(18):
        u,v,y=pieces(tx,tz,ty,nt,group);uu,vv,yy=pieces(vx,vz,vy,nv,group);u,v,y=torch.cat([u,uu]),torch.cat([v,vv]),torch.cat([y,yy]);parts=[]
        for start in range(0,len(u),256):parts.append(y[start:start+256]-cross_response(u[start:start+256],v[start:start+256],mu,c,q,L,R,D,p))
        responses.append(torch.cat(parts).reshape(1024,4,1152))
    responses=torch.stack(responses);np.save(STEM.with_name(STEM.name+'_responses.npy'),responses.cpu().numpy());rng=np.random.default_rng(cfg['partition_seed']);results=[];partitions=[]
    for rep in range(cfg['replicates']):
        perm=rng.permutation(1024);partitions.append(perm.tolist());assert np.array_equal(np.sort(perm),np.arange(1024));covs=[]
        for b in range(4):
            y=responses[:,torch.tensor(perm[b*256:(b+1)*256],device='cuda')].reshape(18,1024,1152).double();covs.append(y.transpose(1,2)@y)
        covs=torch.stack(covs);values={str(k):np.zeros((4,4,18)) for k in cfg['output_ranks']}
        for fit in range(4):
            for g in range(18):
                _,U=torch.linalg.eigh(covs[fit,g]);U=U.flip(1)
                for k in cfg['output_ranks']:
                    v=U[:,:k]
                    for test in range(4):values[str(k)][fit,test,g]=float(torch.einsum('ik,ij,jk->',v,covs[test,g],v)/covs[test,g].trace())
        results.append({k:v.tolist() for k,v in values.items()});print('replicate',rep,'complete',flush=True)
    old=json.loads((ART/'bilin18_l17_output_block_transfer.json').read_text())['results'];summary={}
    for k in cfg['output_ranks']:
        obs=np.array([[np.mean(v) for v in old[f'block{i}']['specialist'][str(k)]] for i in range(4)]);sim=[]
        for rep in results:
            m=np.array(rep[str(k)]).mean(2);inside=float(np.diag(m).mean());outside=float(m[~np.eye(4,dtype=bool)].mean());sim.append({'within':inside,'cross':outside,'gap':inside-outside})
        summary[str(k)]={'observed_within':float(np.diag(obs).mean()),'observed_cross':float(obs[~np.eye(4,dtype=bool)].mean()),'observed_gap':float(np.diag(obs).mean()-obs[~np.eye(4,dtype=bool)].mean()),'repartitions':sim}
    gates={'boundary_effect':all(v['observed_gap']-max(z['gap'] for z in v['repartitions'])>=.05 for v in summary.values())};STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'partitions':partitions,'results':results,'summary':summary,'gates':gates},indent=2)+'\n');print('FINISHED',summary,gates,flush=True)
if __name__=='__main__':main()

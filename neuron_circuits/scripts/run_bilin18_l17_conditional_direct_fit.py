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
    curves={};candidates={};selected={};gen=torch.Generator(device='cuda')
    for initial in cfg['initializations']:
        for lr in cfg['learning_rates']:
            key=f'{initial}_lr{lr}';w=torch.nn.Parameter(base[initial].clone());opt=torch.optim.Adam([w],lr=lr);gen.manual_seed(cfg['seed']);best=-float('inf');curves[key]=[]
            for step in range(cfg['steps']+1):
                if step in cfg['checkpoints']:
                    q=partner_q(w,c);val=score(q,vx,vz,vy);curves[key].append({'step':step,'validation_fidelity':val});print(key,step,val,flush=True)
                    if val>best:best=val;candidates[key]=q.detach().clone();selected[key]={'step':step,'validation_fidelity':val}
                if step==cfg['steps']:break
                ix=torch.randint(len(tx),(cfg['train_batch'],),generator=gen,device='cuda');q=partner_q(w,c);pred=cross_response(tx[ix],tz[ix],mu,c,q,L,R,D,p);loss=(pred-ty[ix]).square().sum(1).mean()/normalizer;opt.zero_grad();loss.backward();opt.step()
                with torch.no_grad():w.copy_(partner_q(w,c))
    winner=max(selected,key=lambda k:selected[k]['validation_fidelity']);qs={**base,'optimized':candidates[winner]};audits={n:{'train':score(q,tx,tz,ty),'validation':score(q,vx,vz,vy),'overlap':float((c.T@q).abs().max()),'orthogonality':float((q.T@q-torch.eye(cfg.get('rank',32),device='cuda')).abs().max())} for n,q in qs.items()};np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),core=c.cpu().numpy(),mean=mu.cpu().numpy(),**{k:v.detach().cpu().numpy() for k,v in {**qs,**candidates}.items()});STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'config':cfg,'curves':curves,'selected':selected,'winner':winner,'audits':audits},indent=2));print('FROZEN',winner,audits,flush=True)
    half=len(tx)//2;vhalf=len(vx)//2
    breakdown={n:{'actual_train':score(q,tx[:half],tz[:half],ty[:half]),'angle_train':score(q,tx[half:],tz[half:],ty[half:]),'actual_validation':score(q,vx[:vhalf],vz[:vhalf],vy[:vhalf]),'angle_validation':score(q,vx[vhalf:],vz[vhalf:],vy[vhalf:])} for n,q in qs.items()}
    selection_path=STEM.with_name(STEM.name+'_selection.json');rec=json.loads(selection_path.read_text());rec['payload_breakdown']=breakdown;selection_path.write_text(json.dumps(rec,indent=2));print('BREAKDOWN',breakdown,flush=True)
if __name__=='__main__':main()

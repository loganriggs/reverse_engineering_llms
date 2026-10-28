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
STEM=ART/'bilin18_l17_direct_partner_fit'
def partner_q(w,core):return torch.linalg.qr(w-core@(core.T@w),mode='reduced').Q

def main():
    global STEM
    if len(sys.argv)>1:STEM=ART/sys.argv[1]
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float()
    L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');f=load_program(ART/'bilin18_l17_partner_compression_bases.npz','cuda');c=torch.linalg.qr(f['natural_core']).Q;mu=f['mean'];base={n:partner_q(f['natural_'+n][:,:cfg.get('rank',32)],c) for n in ['tensor','weighted','random0']}
    with torch.no_grad():
        pre,heads,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':1024});x=F.rms_norm(pre,(1152,));zs=[F.rms_norm(pre-heads[:,h],(1152,)) for h in cfg['heads']];cut=768*4;tx=torch.cat([x[:cut] for _ in zs]);tz=torch.cat([z[:cut] for z in zs]);vx=torch.cat([x[cut:] for _ in zs]);vz=torch.cat([z[cut:] for z in zs])
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
    with torch.no_grad():
        pre,heads,replay=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,)).double();L,R,D=[v.double() for v in [L,R,D]];p={k:v.double() for k,v in p.items()};c=torch.linalg.qr(c.double()).Q;mu=mu.double();qs={n:partner_q(q.double(),c) for n,q in qs.items()};err=lambda z:B(z,z,L,R,D)-forward_normalized(z,p);e0=err(x);rows={}
        for h in cfg['heads']:
            for scale in cfg['scales']:
                z=F.rms_norm(pre-scale*heads[:,h],(1152,)).double();target=err(z)-e0;t=quadratic_terms(x,z,mu,c,L,R,D,p);lin=mean_linear(z-x,mu,L,R,D,p);prefix=f's{scale}_h{h}';rows[prefix+'_cross_energy']=t[:,1].square().sum(-1).reshape(128,4).sum(1).cpu().numpy();rows[prefix+'_total_energy']=target.square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
                for n,q in qs.items():
                    pred=cross_response(x,z,mu,c,q,L,R,D,p)
                    for metric,diff in [('cross',pred-t[:,1]),('total',lin+t[:,0]+pred-target)]:rows[f'{prefix}_{n}_{metric}']=(diff.square().sum(-1)).reshape(128,4).sum(1).cpu().numpy()
            print('test head',h,'complete',flush=True)
    ix=np.random.default_rng(cfg['seed']+1).integers(128,size=(2000,128));results={};contrasts={}
    for scale in cfg['scales']:
        prefix=f's{scale}';total=lambda key:sum(rows[f'{prefix}_h{h}_{key}'] for h in cfg['heads']);results[prefix]={};contrasts[prefix]={}
        for metric in ['cross','total']:
            energy=total(metric+'_energy')
            for n in qs:
                e=total(n+'_'+metric);results[prefix][n+'_'+metric]={'fidelity':float(1-e.sum()/energy.sum()),'document_bootstrap95':np.quantile(1-e[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
            d=total('weighted_'+metric)-total('optimized_'+metric);contrasts[prefix][metric]={'gain':float(d.sum()/energy.sum()),'paired95':np.quantile(d[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
    gates={'direct_gain':all(contrasts[f's{s}']['cross']['gain']>=.1 and contrasts[f's{s}']['cross']['paired95'][0]>0 for s in cfg['scales']),'compact80':all(results[f's{s}']['optimized_cross']['fidelity']>=.8 for s in cfg['scales']),'total_gain':all(contrasts[f's{s}']['total']['gain']>=.05 and contrasts[f's{s}']['total']['paired95'][0]>0 for s in cfg['scales'])};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'winner':winner,'audits':audits,'results':results,'contrasts':contrasts,'gates':gates,'head_replay':replay},indent=2));print('GATES',gates,flush=True)
if __name__=='__main__':main()

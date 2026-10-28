"""H274 fixed-rank actual/angle objective factorial."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_partner_compression import cross_response
from run_bilin18_l17_core_interaction_factorial import quadratic_terms
from run_bilin18_l17_residual_term_accounting import mean_linear
from run_bilin18_l17_direct_partner_fit import partner_q
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_partner_objective_mixture'
def combined_loss(errors,normalizers,weights):return sum(weights[k]*errors[k]/normalizers[k] for k in weights if weights[k])
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[17];L,R,D=[m.weight.detach() for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');f=load_program(ART/'bilin18_l17_partner_compression_bases.npz','cuda');c=torch.linalg.qr(f['natural_core']).Q;mu=f['mean'];base={n:partner_q(f['natural_'+n][:,:32],c) for n in ['weighted','random0']};weights={'actual':{'actual':1.,'angle':0.},'angle':{'actual':0.,'angle':1.},'mixture':{'actual':.5,'angle':.5}};gen=torch.Generator(device='cuda').manual_seed(cfg['payload_seed'])
    with torch.no_grad():
        pre,heads,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':1024});x=F.rms_norm(pre,(1152,));zs={'actual':[],'angle':[]}
        for h in cfg['heads']:
            actual=heads[:,h];source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;angle=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,generator=gen,device='cuda'))
            for n,pay in [('actual',actual),('angle',angle)]:zs[n].append(F.rms_norm(pre-pay,(1152,)))
        cut=768*4;tx=torch.cat([x[:cut] for _ in cfg['heads']]);vx=torch.cat([x[cut:] for _ in cfg['heads']]);tz={n:torch.cat([z[:cut] for z in zz]) for n,zz in zs.items()};vz={n:torch.cat([z[cut:] for z in zz]) for n,zz in zs.items()}
        def targets(xx,zz):return torch.cat([quadratic_terms(xx[i:i+256],zz[i:i+256],mu,c,L,R,D,p)[:,1] for i in range(0,len(xx),256)])
        ty={n:targets(tx,z) for n,z in tz.items()};vy={n:targets(vx,z) for n,z in vz.items()};norm={n:v.square().sum(1).mean() for n,v in ty.items()};print('targets cached',flush=True)
    @torch.no_grad()
    def metrics(q,xx,zz,yy):
        es={}
        for n in zz:
            e=0.
            for i in range(0,len(xx),256):e+=(cross_response(xx[i:i+256],zz[n][i:i+256],mu,c,q,L,R,D,p)-yy[n][i:i+256]).square().sum().double()
            es[n]=float(e/len(xx))
        fs={n:float(1-es[n]/yy[n].square().sum(1).mean()) for n in zz};return es,fs
    curves={};candidates={};selected={};qs={'weighted':base['weighted']};winners={}
    for objective,ow in weights.items():
        for initial in cfg['initializations']:
            for lr in cfg['learning_rates']:
                key=f'{objective}_{initial}_lr{lr}';w=torch.nn.Parameter(base[initial].clone());opt=torch.optim.Adam([w],lr=lr);gen.manual_seed(cfg['seed']);best=float('inf');curves[key]=[]
                for step in range(cfg['steps']+1):
                    if step in cfg['checkpoints']:
                        q=partner_q(w,c);es,fs=metrics(q,vx,vz,vy);loss=float(combined_loss(es,norm,ow));curves[key].append({'step':step,'validation_loss':loss,'validation_fidelities':fs});print(key,step,fs,flush=True)
                        if loss<best:best=loss;candidates[key]=q.detach().clone();selected[key]={'step':step,'validation_loss':loss,'validation_fidelities':fs}
                    if step==cfg['steps']:break
                    ix=torch.randint(len(tx),(cfg['train_batch'],),generator=gen,device='cuda');q=partner_q(w,c);errors={};start=0
                    for n,weight in ow.items():
                        if not weight:continue
                        size=int(cfg['train_batch']*weight);ids=ix[start:start+size];start+=size;pred=cross_response(tx[ids],tz[n][ids],mu,c,q,L,R,D,p);errors[n]=(pred-ty[n][ids]).square().sum(1).mean()
                    loss=combined_loss(errors,norm,ow);opt.zero_grad();loss.backward();opt.step()
                    with torch.no_grad():w.copy_(partner_q(w,c))
        winner=min([k for k in selected if k.startswith(objective+'_')],key=lambda k:selected[k]['validation_loss']);winners[objective]=winner;qs[objective]=candidates[winner]
    audits={n:{'train':metrics(q,tx,tz,ty)[1],'validation':metrics(q,vx,vz,vy)[1]} for n,q in qs.items()};np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),core=c.cpu().numpy(),mean=mu.cpu().numpy(),**{k:v.detach().cpu().numpy() for k,v in {**qs,**candidates}.items()});STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'config':cfg,'curves':curves,'selected':selected,'winners':winners,'audits':audits,'train_energy_normalizers':{k:float(v) for k,v in norm.items()}},indent=2));print('FROZEN',winners,flush=True)
    with torch.no_grad():
        pre,heads,replay=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,)).double();L,R,D=[v.double() for v in [L,R,D]];p={k:v.double() for k,v in p.items()};c=torch.linalg.qr(c.double()).Q;mu=mu.double();qs={n:partner_q(q.double(),c) for n,q in qs.items()};err=lambda z:B(z,z,L,R,D)-forward_normalized(z,p);e0=err(x);rows={};gen.manual_seed(cfg['test_payload_seed'])
        for h in cfg['heads']:
            actual=heads[:,h];source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;angle=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,generator=gen,device='cuda'))
            for label,pay in [('actual',actual),('angle',angle)]:
                for scale in cfg['scales']:
                    z=F.rms_norm(pre-scale*pay,(1152,)).double();target=err(z)-e0;t=quadratic_terms(x,z,mu,c,L,R,D,p);lin=mean_linear(z-x,mu,L,R,D,p);prefix=f'{label}_s{scale}_h{h}'
                    for metric,v in [('cross_energy',t[:,1].square().sum(-1)),('total_energy',target.square().sum(-1))]:rows[prefix+'_'+metric]=v.reshape(128,4).sum(1).cpu().numpy()
                    for n,q in qs.items():
                        pred=cross_response(x,z,mu,c,q,L,R,D,p)
                        for metric,diff in [('cross',pred-t[:,1]),('total',lin+t[:,0]+pred-target)]:rows[f'{prefix}_{n}_{metric}']=diff.square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
            print('test head',h,'complete',flush=True)
    ix=np.random.default_rng(cfg['seed']+1).integers(128,size=(2000,128));results={};contrasts={}
    for label in ['actual','angle']:
        for scale in cfg['scales']:
            prefix=f'{label}_s{scale}';total=lambda key:sum(rows[f'{prefix}_h{h}_{key}'] for h in cfg['heads']);results[prefix]={};contrasts[prefix]={}
            for metric in ['cross','total']:
                energy=total(metric+'_energy')
                for n in qs:
                    e=total(n+'_'+metric);results[prefix][n+'_'+metric]={'fidelity':float(1-e.sum()/energy.sum()),'document_bootstrap95':np.quantile(1-e[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
                for other in ['actual','angle','weighted']:
                    d=total(other+'_'+metric)-total('mixture_'+metric);contrasts[prefix][other+'_'+metric]={'gain':float(d.sum()/energy.sum()),'paired95':np.quantile(d[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
    gates={'angle_rescue':all(contrasts[f'angle_s{s}']['actual_cross']['gain']>=.1 and contrasts[f'angle_s{s}']['actual_cross']['paired95'][0]>0 for s in cfg['scales']),'actual_retention':all(contrasts[f'actual_s{s}']['actual_cross']['paired95'][0]>-.03 for s in cfg['scales']),'near_specialists':all(contrasts[f'{n}_s{s}'][n+'_cross']['paired95'][0]>-.05 for n in ['actual','angle'] for s in cfg['scales']),'both_half':all(results[f'{n}_s{s}']['mixture_cross']['fidelity']>=.5 for n in ['actual','angle'] for s in cfg['scales'])};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'winners':winners,'audits':audits,'results':results,'contrasts':contrasts,'gates':gates,'head_replay':replay},indent=2));print('GATES',gates,flush=True)
if __name__=='__main__':main()

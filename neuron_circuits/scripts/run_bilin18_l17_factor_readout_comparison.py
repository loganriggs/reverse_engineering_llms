"""H326 native, inherited and refitted readouts on identical native-factor subsets."""
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
STEM=ART/'bilin18_l17_factor_readout_comparison'
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
    path=ART/'bilin18_l17_document_repartition_responses.npy';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['response_sha256'];Y=torch.tensor(np.load(path),device='cuda').reshape(18,4096,1152).double()
    path=ART/'bilin18_l17_shared_kernel_coverage_6144_frozen.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['mixed_sha256'];f=np.load(path);ref=[torch.tensor(f[k],device='cuda') for k in ['ux','vx','uz','vz']];assignment=torch.tensor(f['assignment'],device='cuda');labels=Y[assignment,torch.arange(6144,device='cuda')%3072]
    common=np.load(ART/'bilin18_l17_kernel_confirmation_frozen/common.npz');c=torch.tensor(common['core'],device='cuda').double();q=torch.tensor(common['partner'],device='cuda').double();mu=torch.tensor(common['mean'],device='cuda').double()
    def coords(v):
        w=v.double()-mu;u=w@c;return u,w-u@c.T-(w@q)@q.T
    u,v=coords(x[3072:]);queries=[]
    for z in zs_actual+zs_angle:
        w,t=coords(z[3072:]);queries.append((u,v,w,t))
    path=ART/'bilin18_l17_native_factor_readout_native_frozen.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['native_sha256'];full=np.load(path);ll=L.double();rr=R.double();lc=c.T@ll.T;rc=c.T@rr.T
    def features(z):
        aa,bb,cc,dd=z
        return (cc@lc)*(dd@rr.T)+(cc@rc)*(dd@ll.T)-(aa@lc)*(bb@rr.T)-(aa@rc)*(bb@ll.T)
    for filename,digest in cfg['subset_sha256'].items():assert hashlib.sha256((ART/filename).read_bytes()).hexdigest()==digest
    fullcoef=torch.tensor(full['coef'],device='cuda')/float(full['scale'])**.5;native=D.double().T;old=np.load(ART/'bilin18_l17_factor_cancellation_rows.npz');rows={};replay=0.;models={}
    for method in ['ranked','random']:
      for width in cfg['widths']:
        name=f'{method}{width}';sub=load_program(ART/f'bilin18_l17_native_factor_subsets_{name}_frozen.npz','cuda',torch.float64);models[name]=(sub['ids'].long(),sub['coef']/sub['scale'].sqrt())
    models['full']=(torch.arange(4608,device='cuda'),fullcoef)
    for g,query in enumerate(queries):
      raw=features(query);target=Y[g,3072:];rows[f'g{g}_energy']=target.square().sum(1).reshape(256,4).sum(1).cpu().numpy()
      for name,(ids,refitcoef) in models.items():
        phi=raw[:,ids];preds={'native':phi@native[ids],'inherited':phi@fullcoef[ids],'refit':phi@refitcoef}
        for variant,pred in preds.items():rows[f'{name}_g{g}_{variant}_error']=(pred-target).square().sum(1).reshape(256,4).sum(1).cpu().numpy()
        rows[f'{name}_g{g}_correction_energy']=(preds['inherited']-preds['native']).square().sum(1).reshape(256,4).sum(1).cpu().numpy()
        if name!='full':
          for variant,oldkey in [('inherited','frozen_error'),('refit','refit_error')]:
            before=old[f'{name}_g{g}_{oldkey}'];after=rows[f'{name}_g{g}_{variant}_error'];replay=max(replay,float(np.linalg.norm(after-before)/np.linalg.norm(before)))
    assert replay<1e-8
    boot=np.random.default_rng(cfg['bootstrap_seed']).integers(128,size=(2000,128));summary={};contrasts={}
    for name in models:
      summary[name]={};contrasts[name]={}
      for family,groups in [('actual',range(9)),('angle',range(9,18))]:
        energy=sum(rows[f'g{g}_energy'] for g in groups)[128:];errs={v:sum(rows[f'{name}_g{g}_{v}_error'] for g in groups)[128:] for v in ['native','inherited','refit']};summary[name][family]={v:float(1-er.sum()/energy.sum()) for v,er in errs.items()};summary[name][family]['correction_energy_ratio']=float(sum(rows[f'{name}_g{g}_correction_energy'][128:].sum() for g in groups)/energy.sum());gain=errs['native']-errs['inherited'];contrasts[name][family]={'inherited_gain':float(gain.sum()/energy.sum()),'paired95':np.quantile(gain[boot].sum(1)/energy[boot].sum(1),[.025,.975]).tolist()}
    gates={'inherited1024_gain5':all(v['inherited_gain']>=.05 and v['paired95'][0]>0 for v in contrasts['ranked1024'].values()),'native1024_80':all(v['native']>=.8 for v in summary['ranked1024'].values())}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'contrasts':contrasts,'max_previous_error_replay_relative':replay,'gates':gates},indent=2)+'\n');print('FINISHED',summary,contrasts['ranked1024'],gates,flush=True)
if __name__=='__main__':main()

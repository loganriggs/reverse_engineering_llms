"""H325 retained/omitted native-factor accounting with frozen readouts."""
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
STEM=ART/'bilin18_l17_factor_cancellation'
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
    coef=torch.tensor(full['coef'],device='cuda');scale=float(full['scale']);previous=json.loads((ART/'bilin18_l17_native_factor_subsets.json').read_text());previousrows=np.load(ART/'bilin18_l17_native_factor_subsets_rows.npz');rows={};audits={'max_energy_identity_relative':0.,'max_refit_error_replay_relative':0.}
    frozen={}
    for method in ['ranked','random']:
      for width in cfg['widths']:
        name=f'{method}{width}';frozen[name]=load_program(ART/f'bilin18_l17_native_factor_subsets_{name}_frozen.npz','cuda',torch.float64)
    for g,query in enumerate(queries):
      raw=features(query);prediction=raw/scale**.5@coef;target=Y[g,3072:];residual=target-prediction
      for name,sub in frozen.items():
        ids=sub['ids'].long();retained=raw[:,ids]/scale**.5@coef[ids];omitted=prediction-retained;refit=raw[:,ids]/sub['scale'].sqrt()@sub['coef'];correction=refit-retained
        vals={'target_energy':target.square().sum(1),'retained_energy':retained.square().sum(1),'omitted_energy':omitted.square().sum(1),'retained_omitted_dot':(retained*omitted).sum(1),'prediction_energy':prediction.square().sum(1),'full_error':residual.square().sum(1),'frozen_error':(retained-target).square().sum(1),'refit_error':(refit-target).square().sum(1),'omitted_recovery_error':(correction-omitted).square().sum(1)}
        identity=vals['retained_energy']+vals['omitted_energy']+2*vals['retained_omitted_dot']-vals['prediction_energy'];audits['max_energy_identity_relative']=max(audits['max_energy_identity_relative'],float(identity.abs().max()/vals['prediction_energy'].max()))
        for key,value in vals.items():rows[f'{name}_g{g}_{key}']=value.reshape(256,4).sum(1).cpu().numpy()
        old=previousrows[f'{name}_g{g}_a{previous["selected"][name]}_error'];new=rows[f'{name}_g{g}_refit_error'];audits['max_refit_error_replay_relative']=max(audits['max_refit_error_replay_relative'],float(np.linalg.norm(old-new)/np.linalg.norm(old)))
    assert audits['max_energy_identity_relative']<1e-10;assert audits['max_refit_error_replay_relative']<1e-8
    summary={}
    for name in frozen:
      summary[name]={}
      for family,groups in [('actual',range(9)),('angle',range(9,18))]:
        total={key:sum(rows[f'{name}_g{g}_{key}'][128:].sum() for g in groups) for key in vals};e=total['target_energy'];summary[name][family]={'retained_energy_ratio':float(total['retained_energy']/e),'omitted_energy_ratio':float(total['omitted_energy']/e),'prediction_energy_ratio':float(total['prediction_energy']/e),'component_cosine':float(total['retained_omitted_dot']/np.sqrt(total['retained_energy']*total['omitted_energy'])),'frozen_fidelity':float(1-total['frozen_error']/e),'refit_fidelity':float(1-total['refit_error']/e),'full_fidelity':float(1-total['full_error']/e),'refit_gain':float((total['frozen_error']-total['refit_error'])/e),'omitted_recovery':float(1-total['omitted_recovery_error']/total['omitted_energy'])}
    gates={'ranked1024_cancellation':all(v['component_cosine']<=-.1 for v in summary['ranked1024'].values()),'ranked1024_omitted_recovery50':all(v['omitted_recovery']>=.5 for v in summary['ranked1024'].values())}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'audits':audits,'gates':gates},indent=2)+'\n');print('FINISHED',summary,'audits',audits,'gates',gates,flush=True)
if __name__=='__main__':main()

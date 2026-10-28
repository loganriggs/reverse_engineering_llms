"""H324 native-factor subsets selected from training contributions versus random."""
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
STEM=ART/'bilin18_l17_native_factor_subsets'
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
    raw=features(ref);query_features=[features(z) for z in queries];fullcoef=torch.tensor(full['coef'],device='cuda');contribution=raw.square().mean(0)/float(full['scale'])*fullcoef.square().sum(1);ranked=torch.argsort(contribution,descending=True);gen=torch.Generator(device='cuda').manual_seed(cfg['subset_seed']);random=torch.randperm(4608,generator=gen,device='cuda');orders={'ranked':ranked,'random':random};rows={};results={};selected={};audits={};replay=None
    np.savez_compressed(STEM.with_name(STEM.name+'_selection.npz'),ranked=ranked.cpu().numpy(),random=random.cpu().numpy(),contribution=contribution.cpu().numpy())
    for method,order in orders.items():
      for width in cfg['widths']:
        if method=='random' and width==4608:continue
        name=f'{method}{width}';ids=order[:width];raws=raw[:,ids];scale=raws.square().sum(1).mean();design=raws/scale.sqrt();gram=design.T@design;ev,U=torch.linalg.eigh((gram+gram.T)/2);minimum=float(ev.min());ev=ev.clamp_min(0);rhs=U.T@(design.T@labels)
        for g,rawq in enumerate(query_features):
            left=(rawq[:,ids]/scale.sqrt())@U;target=Y[g,3072:];rows[f'g{g}_energy']=target.square().sum(1).reshape(256,4).sum(1).cpu().numpy()
            for alpha in cfg['ridge']:
                pred=(left/(ev+alpha)[None,:])@rhs;rows[f'{name}_g{g}_a{alpha}_error']=(pred-target).square().sum(1).reshape(256,4).sum(1).cpu().numpy()
        results[name]={}
        for family,groups in [('actual',range(9)),('angle',range(9,18))]:
            energy=sum(rows[f'g{g}_energy'] for g in groups);results[name][family]={}
            for alpha in cfg['ridge']:
                err=sum(rows[f'{name}_g{g}_a{alpha}_error'] for g in groups);results[name][family][str(alpha)]={split:float(1-err[sl].sum()/energy[sl].sum()) for split,sl in [('calibration',slice(0,128)),('evaluation',slice(128,None))]}
        selected[name]=max([str(v) for v in cfg['ridge']],key=lambda k:min(results[name][fam][k]['calibration'] for fam in ['actual','angle']));alpha=float(selected[name]);coef=U@(rhs/(ev+alpha)[:,None]);residual=float(((gram+alpha*torch.eye(width,device='cuda',dtype=torch.float64))@coef-design.T@labels).norm()/(design.T@labels).norm());assert residual<1e-7
        audits[name]={'min_eigenvalue':minimum,'normal_equation_relative':residual,'retained_contribution_fraction':float(contribution[ids].sum()/contribution.sum()),'stored_scalars':int(3*width*1152+c.numel()+q.numel()+mu.numel()+2)}
        if width==4608:
            replay=float((coef-fullcoef[ids]).norm()/fullcoef.norm());assert replay<1e-7
        np.savez_compressed(STEM.with_name(STEM.name+'_'+name+'_frozen.npz'),left=ll[ids].cpu().numpy(),right=rr[ids].cpu().numpy(),core=c.cpu().numpy(),partner=q.cpu().numpy(),mean=mu.cpu().numpy(),coef=coef.cpu().numpy(),scale=np.array(float(scale)),ridge=np.array(alpha),ids=ids.cpu().numpy())
        print(name,'selected',selected[name],{f:results[name][f][selected[name]]['evaluation'] for f in ['actual','angle']},flush=True)
    bootstrap=np.random.default_rng(cfg['bootstrap_seed']).integers(128,size=(2000,128));contrasts={}
    for width in cfg['widths']:
      if width==4608:continue
      contrasts[str(width)]={}
      for family,groups in [('actual',range(9)),('angle',range(9,18))]:
        energy=sum(rows[f'g{g}_energy'] for g in groups)[128:];errors={m:sum(rows[f'{m}{width}_g{g}_a{selected[f"{m}{width}"]}_error'] for g in groups)[128:] for m in orders};gain=errors['random']-errors['ranked'];contrasts[str(width)][family]={'gain':float(gain.sum()/energy.sum()),'paired95':np.quantile(gain[bootstrap].sum(1)/energy[bootstrap].sum(1),[.025,.975]).tolist()}
    gates={'ranked1024_80':all(results['ranked1024'][f][selected['ranked1024']]['evaluation']>=.8 for f in ['actual','angle']),'ranked1024_gain10':all(v['gain']>=.1 and v['paired95'][0]>0 for v in contrasts['1024'].values())}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'selected':selected,'audits':audits,'contrasts':contrasts,'full_coefficient_replay_relative':replay,'gates':gates},indent=2)+'\n');print('FINISHED',gates,flush=True)
if __name__=='__main__':main()

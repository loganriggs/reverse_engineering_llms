import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_canonical_matched_oracle'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();old=np.load(ART/'bilin18_l17_canonical_input_expansion_rows.npz');p=torch.tensor(old['pre'][-512:],device='cuda');h=torch.tensor(np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['sources'][0],device='cuda').float()[np.arange(512)%64];z=np.load(ART/'bilin18_l17_canonical_width_budget_rank128_models.npz');z={k:torch.tensor(z[k],device='cuda') for k in z.files}
 def native(x):return ((x@L.T)*(x@R.T))@D.T+bias
 def lp(s):return (30*torch.tanh(F.rms_norm(s,(1152,))@W.T/30)).double().log_softmax(-1)
 learned=np.load(ART/'bilin18_l17_canonical_input_expansion_models.npz');selection=json.loads((ART/'bilin18_l17_canonical_input_expansion_selection.json').read_text())['selected']['r256'];C=torch.tensor(learned[selection+'_coefficients'],device='cuda');B=torch.tensor(learned['basis'],device='cuda');E=torch.tensor(learned['expanded_kernel_encoder'],device='cuda')
 def predict(x):
  k=(x.double()@z['support'].T/1152).square();return (k-z['kernel_mean'][None]-k.mean(1,keepdim=True)+z['kernel_grand'])@E
 x0=F.rms_norm(p,(1152,));y0=native(x0);s=p+y0;v0=predict(x0);saved={};summary={};audit={};audit['orthogonal_error']=float((B@B.T-torch.eye(16,device='cuda')).abs().max());audit['normal_equation_relative_max']=0.;assert audit['orthogonal_error']<1e-10
 for name,alpha in [('negative',-1),('positive',1)]:
  xx=F.rms_norm(p-alpha*h,(1152,));yy=native(xx);v=(z['scale']*(((predict(xx)-v0)@C)@B)).float();u=yy-y0;rows=[];local=[];coefs=[];spectra=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);ss=s[sl];vv=v[sl].double();uu=u[sl].double();th=torch.tanh(F.rms_norm(ss,(1152,)).double()@W.double().T/30);prob0=(30*th).softmax(-1)
   def tangent(t):
    q=(native_drms(ss.double(),t)@W.double().T)*(1-th.square());return q-(prob0*q).sum(-1,keepdim=True)
   tv=tangent(vv);tu=tangent(uu);tb=torch.stack([tangent(b.expand_as(vv)) for b in B],1);G=torch.einsum('biv,bv,bjv->bij',tb,prob0,tb);rhs=torch.einsum('biv,bv,bv->bi',tb,prob0,tu);ev,Q=torch.linalg.eigh(G);keep=ev>ev[:,-1:]*cfg['eigenvalue_relative_cutoff'];inv=torch.where(keep,ev.reciprocal(),0);c=torch.einsum('bij,bj->bi',Q,inv*torch.einsum('bji,bj->bi',Q,rhs));res=(G@c[:,:,None]).squeeze(-1)-rhs;audit['normal_equation_relative_max']=max(audit['normal_equation_relative_max'],float((res.norm(dim=-1)/rhs.norm(dim=-1)).max()));rawc=uu@B.T;vvF=(prob0*tv.square()).sum(-1);uvF=(prob0*tv*tu).sum(-1);uuF=(prob0*tu.square()).sum(-1);gain=uvF/vvF;deltas=[vv,gain[:,None]*vv,rawc@B,c@B];truth=lp(p[sl]+yy[sl]);pr=truth.exp();effect=(pr*(truth-lp(ss))).sum(-1);errs=[];loc=[]
   for delta in deltas:
    q=lp(ss+delta.float());errs.append(torch.stack([(pr*(truth-q)).sum(-1),(delta.float().double()-uu).square().sum(-1)],-1));td=tangent(delta);loc.append((prob0*(td-tu).square()).sum(-1))
   loc=torch.stack(loc,-1);assert torch.all(loc[:,3]<=loc[:,1]+1e-7*uuF+1e-12);rows.append(torch.stack(errs,1).cpu().numpy());local.append(torch.cat([loc,uuF[:,None],effect[:,None],uu.square().sum(-1)[:,None]],-1).cpu().numpy());coefs.append(torch.stack([rawc,c],1).cpu().numpy());spectra.append(ev.cpu().numpy())
  rr=np.concatenate(rows);ll=np.concatenate(local);cc=np.concatenate(coefs);ee=np.concatenate(spectra);saved[name+'_errors']=rr;saved[name+'_local']=ll;saved[name+'_coefficients']=cc;saved[name+'_eigenvalues']=ee
  sl=slice(0,512) if name=='negative' else slice(512,1024);ref=old['confirmation_r256'][sl];assert np.allclose(rr[:,0],ref,rtol=3e-5,atol=1e-8);audit[name+'_KL_replay_maxabs']=float(np.abs(rr[:,0,0]-ref[:,0]).max())
  summary[name]={m:{'KL_recovery':float(1-rr[:,k,0].sum()/ll[:,5].sum()),'raw_recovery':float(1-rr[:,k,1].sum()/ll[:,6].sum()),'local_recovery':float(1-ll[:,k].sum()/ll[:,4].sum()),'worse_finite_KL_than_original_fraction':float(np.mean(rr[:,k,0]>rr[:,0,0]))} for k,m in enumerate(cfg['methods'])};summary[name]['min_effective_rank']=int((ee>ee[:,-1:]*cfg['eigenvalue_relative_cutoff']).sum(-1).min());summary[name]['condition_quantiles']=np.quantile(ee[:,-1]/ee[:,0],[0,.5,.9,1]).tolist();print(name,summary[name],flush=True)
 assert audit['normal_equation_relative_max']<1e-6;assert all(np.isfinite(v).all() for v in saved.values());saved['output_basis']=B.cpu().numpy();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H480','config':cfg,'summary':summary,'audits':audit,'local_columns':cfg['methods']+['native_Fisher_energy','effect_KL','raw_effect_energy'],'coefficient_methods':['raw','fisher'],'unique_completed_documents':15808},indent=2)+'\n')
if __name__=='__main__':main()

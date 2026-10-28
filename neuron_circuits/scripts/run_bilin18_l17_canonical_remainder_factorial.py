import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_canonical_remainder_factorial'
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
 x0=F.rms_norm(p,(1152,));y0=native(x0);s=p+y0;v0=predict(x0);saved={};summary={};audit={'compensation_identity_relative_max':0.,'remainder_orthogonality_relative_max':0.,'Fisher_residual_orthogonality_relative_max':0.};prior=np.load(ART/'bilin18_l17_canonical_matched_oracle_rows.npz');Wd=W.double()
 for name,alpha in [('negative',-1),('positive',1)]:
  xx=F.rms_norm(p-alpha*h,(1152,));yy=native(xx);v=(z['scale']*(((predict(xx)-v0)@C)@B)).float();u=yy-y0;coeff=torch.tensor(prior[name+'_coefficients'],device='cuda');rows=[];local=[];parts=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);ss=s[sl];vv=v[sl].double();uu=u[sl].double();raw=coeff[sl,0]@B;comp=(coeff[sl,1]-coeff[sl,0])@B;perp=uu-raw;th=torch.tanh(F.rms_norm(ss,(1152,)).double()@Wd.T/30);prob0=(30*th).softmax(-1)
   def tangent(t):
    q=(native_drms(ss.double(),t)@Wd.T)*(1-th.square());return q-(prob0*q).sum(-1,keepdim=True)
   tu=tangent(uu);te=tangent(vv-raw);tp=tangent(perp);tc=tangent(comp);tb=torch.stack([tangent(b.expand_as(vv)) for b in B],1);rhs=torch.einsum('biv,bv,bv->bi',tb,prob0,tp);G=torch.einsum('biv,bv,bjv->bij',tb,prob0,tb);dc=coeff[sl,1]-coeff[sl,0];res=(G@dc[:,:,None]).squeeze(-1)-rhs;audit['compensation_identity_relative_max']=max(audit['compensation_identity_relative_max'],float((res.norm(dim=-1)/rhs.norm(dim=-1)).max()));audit['remainder_orthogonality_relative_max']=max(audit['remainder_orthogonality_relative_max'],float(((perp@B.T).norm(dim=-1)/uu.norm(dim=-1)).max()));orth=torch.einsum('biv,bv,bv->bi',tb,prob0,tp-tc);audit['Fisher_residual_orthogonality_relative_max']=max(audit['Fisher_residual_orthogonality_relative_max'],float((orth.norm(dim=-1)/rhs.norm(dim=-1)).max()));uuF=(prob0*tu.square()).sum(-1);deltas=[vv,raw,vv+perp,raw+perp,vv+comp,raw+comp];truth=lp(p[sl]+yy[sl]);pr=truth.exp();effect=(pr*(truth-lp(ss))).sum(-1);errs=[];loc=[]
   for delta in deltas:
    q=lp(ss+delta.float());errs.append(torch.stack([(pr*(truth-q)).sum(-1),(delta.float().double()-uu).square().sum(-1)],-1));td=tangent(delta);loc.append((prob0*(td-tu).square()).sum(-1))
   loc=torch.stack(loc,-1);cross=(prob0*te*tp).sum(-1);ee=(prob0*te.square()).sum(-1);pp=(prob0*tp.square()).sum(-1);assert torch.allclose(loc[:,0],ee+pp-2*cross,rtol=1e-7,atol=1e-12);assert torch.allclose(loc[:,4],ee+loc[:,5],rtol=1e-6,atol=1e-12);rows.append(torch.stack(errs,1).cpu().numpy());local.append(torch.cat([loc,uuF[:,None],effect[:,None],uu.square().sum(-1)[:,None]],-1).cpu().numpy());parts.append(torch.stack([ee,pp,cross,(prob0*tc.square()).sum(-1),perp.square().sum(-1)],-1).cpu().numpy())
  rr=np.concatenate(rows);ll=np.concatenate(local);parts=np.concatenate(parts);saved[name+'_errors']=rr;saved[name+'_local']=ll;saved[name+'_parts']=parts
  for k,oldk in [(0,0),(1,2),(5,3)]:assert np.allclose(rr[:,k],prior[name+'_errors'][:,oldk],rtol=3e-5,atol=1e-8)
  summary[name]={m:{'KL_recovery':float(1-rr[:,k,0].sum()/ll[:,7].sum()),'raw_recovery':float(1-rr[:,k,1].sum()/ll[:,8].sum()),'local_recovery':float(1-ll[:,k].sum()/ll[:,6].sum()),'worse_KL_fraction':float(np.mean(rr[:,k,0]>rr[:,0,0]))} for k,m in enumerate(cfg['methods'])};summary[name]['local_parts_over_effect']=dict(zip(['inspan_error','omitted_remainder','cross_inner_product','compensation_energy'],(parts[:,:4].sum(0)/ll[:,6].sum()).tolist()));summary[name]['raw_remainder_fraction']=float(parts[:,4].sum()/ll[:,8].sum());assert abs(1-summary[name]['true_inspan_plus_remainder']['KL_recovery'])<1e-5;print(name,summary[name],flush=True)
 assert all(v<1e-6 for v in audit.values());assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H481','config':cfg,'summary':summary,'audits':audit,'local_columns':cfg['methods']+['native_Fisher_energy','effect_KL','raw_effect_energy'],'part_columns':['inspan_error_Fisher','remainder_Fisher','cross_inner_product','compensation_Fisher','raw_remainder_energy'],'unique_completed_documents':15808},indent=2)+'\n')
if __name__=='__main__':main()

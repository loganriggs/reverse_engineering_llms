import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_canonical_augmented_oracle'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();old=np.load(ART/'bilin18_l17_canonical_remainder_learning_rows.npz');p=torch.tensor(old['pre'][-512:],device='cuda');h=torch.tensor(np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['sources'][0],device='cuda').float()[np.arange(512)%64];z=np.load(ART/'bilin18_l17_canonical_width_budget_rank128_models.npz');z={k:torch.tensor(z[k],device='cuda') for k in z.files}
 def native(x):return ((x@L.T)*(x@R.T))@D.T+bias
 def lp(s):return (30*torch.tanh(F.rms_norm(s,(1152,))@W.T/30)).double().log_softmax(-1)
 learned=np.load(ART/'bilin18_l17_canonical_remainder_learning_models.npz');selection=json.loads((ART/'bilin18_l17_canonical_remainder_learning_selection.json').read_text())['selected'];B=torch.tensor(learned['basis'],device='cuda');E=torch.tensor(learned['expanded_kernel_encoder'],device='cuda');Cs={arm:torch.tensor(learned[n+'_coefficients'],device='cuda') for arm,n in selection.items()};indices={'base':list(range(16)),'pca':list(range(20)),'random':list(range(16))+list(range(20,24))};Wd=W.double()
 def feat(x):
  k=(x.double()@z['support'].T/1152).square();return (k-z['kernel_mean'][None]-k.mean(1,keepdim=True)+z['kernel_grand'])@E
 x0=F.rms_norm(p,(1152,));y0=native(x0);state=p+y0;z0=feat(x0);saved={};summary={};audit={'normal_equation_relative_max':0.,'replay_KL_maxabs':0.};coefs={}
 for sign,alpha in [('negative',-1),('positive',1)]:
  xx=F.rms_norm(p-alpha*h,(1152,));yy=native(xx);u=(yy-y0).double();dz=feat(xx)-z0;pred={arm:(z['scale']*((dz@C)@B)).float() for arm,C in Cs.items()};rows={arm:[] for arm in indices};locals={arm:[] for arm in indices};coeffs={arm:[] for arm in indices};ranks={arm:[] for arm in indices};effects=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);ss=state[sl];uu=u[sl];th=torch.tanh(F.rms_norm(ss,(1152,)).double()@Wd.T/30);prob0=(30*th).softmax(-1)
   def tangent(v):
    q=(native_drms(ss.double(),v)@Wd.T)*(1-th.square());return q-(prob0*q).sum(-1,keepdim=True)
   tu=tangent(uu);tb=torch.stack([tangent(b.expand_as(uu)) for b in B],1);GG=torch.einsum('biv,bv,bjv->bij',tb,prob0,tb);rr=torch.einsum('biv,bv,bv->bi',tb,prob0,tu);truth=lp(p[sl]+yy[sl]);pr=truth.exp();effect=(pr*(truth-lp(ss))).sum(-1);effects.append(torch.stack([effect,uu.square().sum(-1),(prob0*tu.square()).sum(-1)],-1).cpu().numpy());batchlocal={}
   for arm,ix in indices.items():
    bb=B[ix];G=GG[:,ix][:,:,ix];rhs=rr[:,ix];ev,Q=torch.linalg.eigh(G);keep=ev>ev[:,-1:]*cfg['eigenvalue_relative_cutoff'];inv=torch.where(keep,ev.reciprocal(),0);c=torch.einsum('bij,bj->bi',Q,inv*torch.einsum('bji,bj->bi',Q,rhs));res=(G@c[:,:,None]).squeeze(-1)-rhs;audit['normal_equation_relative_max']=max(audit['normal_equation_relative_max'],float((res.norm(dim=-1)/rhs.norm(dim=-1)).max()));rawc=uu@bb.T;errs=[];loc=[]
    for delta in [pred[arm][sl].double(),rawc@bb,c@bb]:
     q=lp(ss+delta.float());errs.append(torch.stack([(pr*(truth-q)).sum(-1),(delta.float().double()-uu).square().sum(-1)],-1));td=tangent(delta);loc.append((prob0*(td-tu).square()).sum(-1))
    rows[arm].append(torch.stack(errs,1).cpu().numpy());ll=torch.stack(loc,1);locals[arm].append(ll.cpu().numpy());batchlocal[arm]=ll[:,2];coeffs[arm].append(torch.stack([rawc,c],1).cpu().numpy());ranks[arm].append(keep.sum(-1).cpu().numpy())
   for arm in ['pca','random']:assert torch.all(batchlocal[arm]<=batchlocal['base']+1e-10)
  ef=np.concatenate(effects);saved[sign+'_effects']=ef;summary[sign]={}
  for arm in indices:
   er=np.concatenate(rows[arm]);lo=np.concatenate(locals[arm]);saved[sign+'_'+arm+'_errors']=er;saved[sign+'_'+arm+'_local']=lo;saved[sign+'_'+arm+'_coefficients']=np.concatenate(coeffs[arm]);saved[sign+'_'+arm+'_rank']=np.concatenate(ranks[arm]);sl=slice(0,512) if sign=='negative' else slice(512,1024);ref=old['confirmation_'+arm][sl];assert np.allclose(er[:,0],ref,rtol=3e-5,atol=1e-8);audit['replay_KL_maxabs']=max(audit['replay_KL_maxabs'],float(np.abs(er[:,0,0]-ref[:,0]).max()));summary[sign][arm]={m:{'KL_recovery':float(1-er[:,k,0].sum()/ef[:,0].sum()),'raw_recovery':float(1-er[:,k,1].sum()/ef[:,1].sum()),'local_recovery':float(1-lo[:,k].sum()/ef[:,2].sum())} for k,m in enumerate(cfg['methods'])};summary[sign][arm]['minimum_Gram_rank']=int(saved[sign+'_'+arm+'_rank'].min())
  print(sign,summary[sign],flush=True)
 assert audit['normal_equation_relative_max']<1e-6;assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H483','config':cfg,'summary':summary,'audits':audit,'unique_completed_documents':16320},indent=2)+'\n')
if __name__=='__main__':main()

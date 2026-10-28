import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_canonical_shared_core'
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();Ld,Rd,Dd=L.double(),R.double(),D.double();z=np.load(ART/'bilin18_l17_structured_crossrank_compiled.npz');b=torch.tensor(z['r64_basis'],device='cuda');Q=torch.tensor(z['r64_Q'],device='cuda');V0=torch.tensor(z['r64_write'],device='cuda');bl=Ld@b;br=Rd@b;M=torch.stack([-Dd@(br[:,i,None]*Ld+bl[:,i,None]*Rd) for i in range(4)]);h=torch.tensor(np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['sources'][0],device='cuda').double();eps=torch.finfo(torch.float32).eps;pa=np.load(ART/'bilin18_l17_canonical_learning_curve_rows.npz')['pre'][:1280]
 def native(p):
  x=F.rms_norm(p,(1152,));return ((x@L.T)*(x@R.T))@D.T+bias
 def lp(s):return (30*torch.tanh(F.rms_norm(s.float(),(1152,))@W.T/30)).double().log_softmax(-1)
 @torch.no_grad()
 def panel(pa,donors):
  p=torch.tensor(pa,device='cuda');pd=p.double();y0=native(p);state=p+y0;w0=y0.double()-bias.double();hh=h[np.arange(len(p))%donors];d0=pd.square().mean(-1)+eps;xs=[];os=[];targets=[];us=[]
  for alpha in [-1,1]:
   aa=alpha*(hh@b);hp=aa@b.T;den=(pd-hp).square().mean(-1)+eps;cross=(torch.einsum('nd,kod->nko',pd,M)*aa[:,:,None]).sum(1)/den[:,None];other=torch.einsum('ni,nj,ijo->no',aa,aa,Q)/den[:,None]+(d0/den-1)[:,None]*w0;yy=native(p-alpha*hh.float());xs.append(cross);os.append(other);targets.append(p+yy);us.append((yy-y0).double())
  pan={'X':torch.cat(xs),'other':torch.cat(os),'state':state.repeat(2,1),'target':torch.cat(targets),'U':torch.cat(us),'n':len(p)};pan['truth']=torch.cat([lp(pan['target'][i:i+16]) for i in range(0,2*len(p),16)]);return pan
 torch.set_grad_enabled(False);test=panel(np.load(ART/'bilin18_l17_structured_metric_subspace_rows.npz')['pre'],64);prior=np.load(ART/'bilin18_l17_canonical_stability_rows.npz');saved={};models={};core_dims={};audit={'projector_decomposition_max':0.,'orthogonality_max':0.}
 for rep in range(4):
  A=torch.tensor(prior[f'{rep}_left_basis'],device='cuda');B=torch.tensor(prior[f'{rep}_right_basis'],device='cuda');U,sv,Vh=torch.linalg.svd(A@B.T);n=int((sv>=cfg['principal_cosine_threshold']).sum());assert 0<n<64;core_dims[str(rep)]=n
  for side,full,rotation in [('left',A,U.T),('right',B,Vh)]:
   rotated=rotation@full;core=rotated[:n];tail=rotated[n:];name=f'{rep}_{side}';models[name]={'full':full,'core':core,'tail':tail};saved[name+'_core_basis']=core.cpu().numpy();saved[name+'_tail_basis']=tail.cpu().numpy();err=float((rotated@rotated.T-torch.eye(64,device='cuda')).abs().max());audit['orthogonality_max']=max(audit['orthogonality_max'],err);err=float((core.T@core+tail.T@tail-full.T@full).abs().max());audit['projector_decomposition_max']=max(audit['projector_decomposition_max'],err)
 effects=[];errors={n:{m:[] for m in ms} for n,ms in models.items()};details={n:[] for n in models}
 for start in range(0,1024,16):
  sl=slice(start,start+16);X=test['X'][sl];other=test['other'][sl];truth=test['truth'][sl];pr=truth.exp();base=lp(test['state'][sl]);effects.append(torch.stack([(pr*(truth-base)).sum(-1),test['U'][sl].square().sum(-1)],-1).cpu().numpy())
  for name,ms in models.items():
   projected={m:(X@V.T)@V for m,V in ms.items()};lps={}
   assert torch.allclose(projected['full'],projected['core']+projected['tail'],rtol=1e-7,atol=1e-7)
   for mode,part in projected.items():
    delta=part+other;q=lp(test['state'][sl]+delta.float());lps[mode]=q;errors[name][mode].append(torch.stack([(pr*(truth-q)).sum(-1),(delta-test['U'][sl]).square().sum(-1)],-1).cpu().numpy())
   full=lps['full'];core=lps['core'];omitKL=(full.exp()*(full-core)).sum(-1);details[name].append(torch.stack([projected['full'].square().sum(-1),projected['core'].square().sum(-1),projected['tail'].square().sum(-1),omitKL],-1).cpu().numpy())
 ef=np.concatenate(effects);saved['effects']=ef;metrics={}
 for name,modes in errors.items():
  de=np.concatenate(details[name]);saved[name+'_details']=de;metrics[name]={};rrs={}
  for mode,rs in modes.items():rrs[mode]=np.concatenate(rs);saved[name+'_'+mode+'_errors']=rrs[mode]
  assert np.allclose(rrs['full'],prior[name+'_errors'],rtol=3e-5,atol=1e-8)
  for sign,sl in [('negative',slice(0,512)),('positive',slice(512,1024))]:
   metrics[name][sign]={m:{'full_KL':float(1-v[sl,0].sum()/ef[sl,0].sum()),'full_raw':float(1-v[sl,1].sum()/ef[sl,1].sum())} for m,v in rrs.items()};metrics[name][sign]['core_cross_energy_fraction']=float(de[sl,1].sum()/de[sl,0].sum());metrics[name][sign]['full_to_core_KL_over_native_effect']=float(de[sl,3].sum()/ef[sl,0].sum())
  print(name,metrics[name],flush=True)
 assert all(v<1e-10 for v in audit.values());assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H493','config':cfg,'core_dimensions':core_dims,'metrics':metrics,'audits':audit,'detail_columns':['full_cross_energy','core_cross_energy','tail_cross_energy','KL_full_model_to_core_model'],'unique_completed_documents':18880},indent=2)+'\n')
if __name__=='__main__':main()

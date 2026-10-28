import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_rank1_rational_response'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 prior=np.load(ART/'bilin18_l17_source_rank1_tail_rows.npz');p=torch.tensor(np.load(ART/'bilin18_l17_canonical_finite_refinement_rows.npz')['pre'],device='cuda');pd=p.double();b=torch.tensor(prior['basis'],device='cuda');h=torch.tensor(prior['source_full'],device='cuda');a0=(h@b)[np.arange(512)%64];h1=torch.tensor(prior['source_rank1'],device='cuda')[np.arange(512)%64].float();model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();Ld,Rd,Dd=L.double(),R.double(),D.double();eps=torch.finfo(torch.float32).eps
 def native(q):
  x=F.rms_norm(q,(1152,));return ((x@L.T)*(x@R.T))@D.T+bias
 def lp(s):return (30*torch.tanh(F.rms_norm(s,(1152,))@W.T/30)).double().log_softmax(-1)
 pl=pd@Ld.T;pr=pd@Rd.T;bl=b@Ld.T;br=b@Rd.T;N0=(pl*pr)@Dd.T;N1=-(pl*br+pr*bl)@Dd.T;N2=(bl*br)@Dd.T;d0=pd.square().mean(-1)+eps;d1=-2*(pd*b).mean(-1);d2=b.square().mean();w0=N0/d0[:,None];V1=N1-w0*d1[:,None];V2=N2[None]-w0*d2;y0=native(p);state=p+y0;summary={};audit={};saved={'V1':V1.cpu().numpy(),'V2':V2.cpu().numpy(),'d0':d0.cpu().numpy(),'d1':d1.cpu().numpy(),'d2':float(d2),'source_amplitude':a0.cpu().numpy()}
 for sign,alpha in [('negative',-1),('positive',1)]:
  a=alpha*a0;den=d0+a*d1+a.square()*d2;first=a[:,None]*V1/den[:,None];second=a.square()[:,None]*V2/den[:,None];exact=first+second;changed=pd-a[:,None]*b;xd=changed/(changed.square().mean(-1,keepdim=True)+eps).sqrt();direct=((xd@Ld.T)*(xd@Rd.T))@Dd.T-w0;err=float((exact-direct).norm()/direct.norm());assert err<1e-8;audit[sign+'_double_identity_relative']=err;yy=native(p-alpha*h1);u=(yy-y0).double();nativeerr=float((exact-u).norm()/u.norm());assert nativeerr<1e-4;audit[sign+'_float_native_delta_relative']=nativeerr
  ds={'exact':exact,'tangent':a[:,None]*V1/d0[:,None],'linear_numerator':first,'frozen_norm_full':(a[:,None]*N1+a.square()[:,None]*N2)/d0[:,None],'frozen_norm_linear':a[:,None]*N1/d0[:,None],'normalization_only':N0*(1/den-1/d0)[:,None]};rows={m:[] for m in ds};effects=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);truth=lp(p[sl]+yy[sl]);prob=truth.exp();base=lp(state[sl]);effects.append(torch.stack([(prob*(truth-base)).sum(-1),u[sl].square().sum(-1)],-1).cpu().numpy())
   for m,delta in ds.items():
    q=lp(state[sl]+delta[sl].float());rows[m].append(torch.stack([(prob*(truth-q)).sum(-1),(delta[sl]-u[sl]).square().sum(-1)],-1).cpu().numpy())
  ef=np.concatenate(effects);assert np.allclose(ef,prior[sign+'_rank1'][:,[1,3]],rtol=3e-5,atol=1e-8);saved[sign+'_effects']=ef;summary[sign]={}
  for m,rs in rows.items():
   rr=np.concatenate(rs);saved[sign+'_'+m]=rr;summary[sign][m]={'KL_recovery_rank1':float(1-rr[:,0].sum()/ef[:,0].sum()),'raw_recovery_rank1':float(1-rr[:,1].sum()/ef[:,1].sum())}
  parts=torch.stack([first.square().sum(-1),second.square().sum(-1),(first*second).sum(-1),(den-d0)/d0],-1).cpu().numpy();saved[sign+'_parts']=parts;summary[sign]['raw_numerator_parts_over_exact']=(parts[:,:3].sum(0)/exact.square().sum().item()).tolist();summary[sign]['relative_denominator_change_quantiles']=np.quantile(parts[:,3],[0,.1,.5,.9,1]).tolist();print(sign,summary[sign],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values() if isinstance(v,np.ndarray));np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H486','config':cfg,'summary':summary,'audits':audit,'unique_completed_documents':16832,'part_columns':['first_numerator_energy','second_numerator_energy','cross_inner_product','relative_denominator_change']},indent=2)+'\n')
if __name__=='__main__':main()

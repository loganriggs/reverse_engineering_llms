"""H368 factorial identification of cross-component interference."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_rank_cross_metric_factorial'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 pre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]]
 x=F.rms_norm(pre,(1152,));state=pre+B(x,x,L,R,D)+model.blocks[17].mlp.Down_bias;baseline=F.rms_norm(state,(1152,))
 frozen=np.load(ART/'bilin18_l17_unembedding_rank_prediction_rows.npz');V=torch.tensor(frozen['eigenvectors'][:,:2],device='cuda');W=model.lm_head.weight.double();assert model.lm_head.bias is None
 energy=np.empty((3,13,4,4,512));identity=0.;null=0.
 old=np.load(ART/'bilin18_l17_rank_cross_curvature_rows.npz')['energy']
 for hi,h in enumerate(cfg['heads']):
  responses=np.load(ART/f'bilin18_l17_rank_prediction_confirmation_h{h}_rows.npz')['responses']
  for ci,response in enumerate(responses):
   target=F.rms_norm(state+torch.tensor(response,device='cuda',dtype=torch.float32),(1152,));delta=(target-baseline).double();u=(delta@V)@V.T;v=delta-u
   for off in range(0,512,16):
    sl=slice(off,off+16);raw=baseline[sl].double()@W.T;du=u[sl]@W.T;dv=v[sl]@W.T;prob=(30*torch.tanh(raw/30)).softmax(-1);cap=1-torch.tanh(raw/30).square()
    for mi,(weighted,capped) in enumerate([(False,False),(False,True),(True,False),(True,True)]):
     ju=du*cap if capped else du;jv=dv*cap if capped else dv
     if weighted:
      ju=ju-(ju*prob).sum(-1,keepdim=True);jv=jv-(jv*prob).sum(-1,keepdim=True);eu=(prob*ju.square()).sum(-1);ev=(prob*jv.square()).sum(-1);cross=2*(prob*ju*jv).sum(-1);total=(prob*(ju+jv).square()).sum(-1)
     else:
      ju=ju-ju.mean(-1,keepdim=True);jv=jv-jv.mean(-1,keepdim=True);eu=ju.square().mean(-1);ev=jv.square().mean(-1);cross=2*(ju*jv).mean(-1);total=(ju+jv).square().mean(-1)
     energy[hi,ci,mi,:,sl]=torch.stack([eu,ev,cross,total]).cpu().numpy();identity=max(identity,float(((eu+ev+cross-total).abs()/total).max()))
     if mi==0:null=max(null,float((cross.abs()/(eu*ev).sqrt()).max()))
  print('head',h,'complete',flush=True)
 replay=float(np.max(np.abs(energy[:,:,3]-old))/np.max(np.abs(old)));assert replay<1e-10 and identity<1e-10 and np.isfinite(energy).all()
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),energy=energy)
 analyze(cfg,energy,{'h367_replay_max_relative_to_global_max':replay,'identity_max_relative':identity,'uniform_uncapped_cross_max_normalized':null})
def analyze(cfg,energy,audit):
 groups=[slice(0,1),slice(1,7),slice(7,13)];ev=energy.reshape(3,13,4,4,128,4)
 def stats(e):
  z=e[:,:,:,2]/e[:,:,:,3];p=.5*((z[:,:,2]-z[:,:,0])+(z[:,:,3]-z[:,:,1]));c=.5*((z[:,:,1]-z[:,:,0])+(z[:,:,3]-z[:,:,2]));share=p/(z[:,:,3]-z[:,:,0]);return z,p,c,share
 em=np.stack([ev[:,g].mean((1,4,5)) for g in groups],1);z,p,c,share=stats(em);rng=np.random.default_rng(cfg['bootstrap_seed']);bz=[];bs=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,128,128);ee=[]
  for g in groups:
   e=ev[:,g];rr=rng.integers(0,e.shape[1],e.shape[1]);ee.append(e[:,rr][:,:,:,:,ix].mean((1,4,5)))
  zz,pp,cc,ss=stats(np.stack(ee,1));bz.append(zz);bs.append(ss)
 bz=np.stack(bz);bs=np.stack(bs);zr=float(np.quantile(np.max(np.abs(bz-z),axis=(1,2,3)),.95));sr=float(np.quantile(np.max(np.abs(bs-share),axis=(1,2)),.95));gates={'uniform_uncapped_orthogonality':audit['uniform_uncapped_cross_max_normalized']<1e-8,'probability_alone_cancellation':bool((z[:,:,2]+zr<0).all()),'probability_majority':bool((share-sr>.5).all())}
 assert np.max(np.abs(p+c-(z[:,:,3]-z[:,:,0])))<1e-12
 j={'config':cfg,'families':['actual','moment','donor'],'metric_order':['uniform_no_cap','uniform_cap','probability_no_cap','probability_cap'],'cross_over_total':z.tolist(),'cross_joint36_ci95':np.stack([z-zr,z+zr],-1).tolist(),'probability_contribution':p.tolist(),'cap_contribution':c.tolist(),'probability_share':share.tolist(),'probability_share_joint9_ci95':np.stack([share-sr,share+sr],-1).tolist(),'mean_energy':em.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),cross=bz,probability_share=bs);print('FINISHED',gates,'shares',share,'cross',z,flush=True)
if __name__=='__main__':main()

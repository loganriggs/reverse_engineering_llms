import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_structured_source_transfer'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();z=np.load(ART/'bilin18_l17_structured_crossrank_compiled.npz');b=torch.tensor(z['r64_basis'],device='cuda');Q=torch.tensor(z['r64_Q'],device='cuda');read=torch.tensor(z['r64_read'],device='cuda');write=torch.tensor(z['r64_write'],device='cuda');old=np.load(ART/'bilin18_l17_structured_metric_subspace_rows.npz');p=torch.tensor(old['pre'],device='cuda');pd=p.double();src=np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz');center=torch.tensor(src['calibration_center'],device='cuda');natural=torch.tensor(src['sources'][0],device='cuda');trace=center.square().sum()/(len(center)-1);wo=model.blocks[17].attn.c_proj.weight.detach().reshape(1152,9,128)[:,6].double();U,sv,_=torch.linalg.svd(wo,full_matrices=False);rank=int((sv>sv[0]*1e-10).sum());U=U[:,:rank];rng=np.random.default_rng(cfg['seed']);gauss=torch.tensor(rng.standard_normal((64,len(center))),device='cuda')@center/(len(center)-1)**.5;iso=torch.tensor(rng.standard_normal((64,rank)),device='cuda')@U.T*(trace/rank).sqrt();sources={'natural':natural,'covariance_gaussian':gauss,'head_isotropic':iso};sourceaudit={};audit={'head_output_rank':rank,'head_span_orthogonality':float((U.T@U-torch.eye(rank,device='cuda')).abs().max()),'calibration_covariance_trace':float(trace),'natural_outside_head_relative':float((natural-(natural@U)@U.T).norm()/natural.norm())};assert audit['head_span_orthogonality']<1e-10 and audit['natural_outside_head_relative']<1e-5
 for name,hh in sources.items():sourceaudit[name]={'mean_energy_over_calibration_expected':float(hh.square().sum(-1).mean()/trace),'source4_energy_fraction':float((hh@b).square().sum()/hh.square().sum()),'norm_quantiles':np.quantile(hh.norm(dim=-1).cpu().numpy(),[0,.1,.5,.9,1]).tolist()}
 np.savez_compressed(STEM.with_name(STEM.name+'_sources.npz'),**{n:v.cpu().numpy() for n,v in sources.items()},head_basis=U.cpu().numpy());print('sources frozen',sourceaudit,flush=True)
 def native(p):
  x=F.rms_norm(p,(1152,));return ((x@L.T)*(x@R.T))@D.T+bias
 def lp(s):return (30*torch.tanh(F.rms_norm(s,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p);state=p+y0;w0=y0.double()-bias.double();eps=torch.finfo(torch.float32).eps;d0=pd.square().mean(-1)+eps;coords=torch.einsum('nd,kdr->nkr',pd,read);saved={};metrics={}
 for name,pool in sources.items():
  hh=pool[np.arange(512)%64];metrics[name]={}
  for sign,alpha in [('negative',-1),('positive',1)]:
   a=alpha*(hh@b);hp=a@b.T;den=(pd-hp).square().mean(-1)+eps;cr=(coords*a[:,:,None]).sum(1)@write;quad=torch.einsum('ni,nj,ijo->no',a,a,Q);delta=(cr+quad)/den[:,None]+(d0/den-1)[:,None]*w0;yf=native(p-alpha*hh.float());yp=native(p-hp.float());uf=(yf-y0).double();up=(yp-y0).double();rows=[]
   for start in range(0,512,16):
    sl=slice(start,start+16);tf=lp(p[sl]+yf[sl]);tp=lp(p[sl]+yp[sl]);pf=tf.exp();pp=tp.exp();base=lp(state[sl]);q=lp(state[sl]+delta[sl].float());rows.append(torch.stack([(pf*(tf-base)).sum(-1),(pp*(tp-base)).sum(-1),(pf*(tf-tp)).sum(-1),(pf*(tf-q)).sum(-1),(pp*(tp-q)).sum(-1),uf[sl].square().sum(-1),(uf[sl]-up[sl]).square().sum(-1),(uf[sl]-delta[sl]).square().sum(-1)],-1).cpu().numpy())
   rr=np.concatenate(rows);saved[name+'_'+sign]=rr;v=rr.sum(0);metrics[name][sign]={'mean_full_effect_KL':float(v[0]/512),'exact_projection_full_KL_recovery':float(1-v[2]/v[0]),'compressed_full_KL_recovery':float(1-v[3]/v[0]),'compressed_projected_KL_recovery':float(1-v[4]/v[1]),'exact_projection_raw_recovery':float(1-v[6]/v[5]),'compressed_full_raw_recovery':float(1-v[7]/v[5]),'projected_effect_over_full':float(v[1]/v[0])}
   if name=='natural':
    sl=slice(0,512) if sign=='negative' else slice(512,1024);ref=old['confirmation_initial'][sl];assert np.allclose(rr[:,[3,7]],ref,rtol=3e-5,atol=1e-8);audit[sign+'_natural_KL_replay_maxabs']=float(abs(rr[:,3]-ref[:,0]).max())
  print(name,metrics[name],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H495','config':cfg,'metrics':metrics,'source_audit':sourceaudit,'audits':audit,'columns':['full_effect_KL','projected_effect_KL','exact_projection_error_KL','compressed_full_error_KL','compressed_projected_error_KL','full_raw_energy','projection_raw_error','compressed_raw_error'],'unique_completed_documents':18880},indent=2)+'\n')
if __name__=='__main__':main()

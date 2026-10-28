import json,hashlib,numpy as np,torch
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_teacher_label_variance'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 def T(x):return torch.tensor(x,device='cuda',dtype=torch.float64)
 data=np.load(ART/'bilin18_l17_label_direction_confirmation_rows.npz');z=np.load(ART/'bilin18_l17_source_sample_curve_models.npz');factor=np.load(ART/'bilin18_l17_objective_gain_alignment_rows.npz');directions=np.load(ART/'bilin18_l17_label_direction_confirmation_directions.npz')['directions'];B=T(z['basis']);mu=T(z['mean']);read=T(factor['read']);write=T(factor['write']);p=torch.tensor(data['pre'],device='cuda');h=T(data['head_writes']).sum(1)-mu;coord=h@B@read;inp=p-(h-coord@write.T).float();targets=torch.tensor(data['targets'],device='cuda')
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();eps=torch.finfo(torch.float32).eps
 def norm(x):return x/(x.square().mean(-1,keepdim=True)+eps).sqrt()
 def dn(x,v):
  s=(x.square().mean(-1,keepdim=True)+eps).sqrt();return v/s-x*(x*v).mean(-1,keepdim=True)/s**3
 def soft(p):
  x=norm(p);state=p+((x@L.T)*(x@R.T))@D.T+bias;return torch.tanh(norm(state)@W.T/30)
 covs=[];obs=[];kls=[];mean_residual=0.
 for start in range(0,len(p),16):
  sl=slice(start,start+16);pp=inp[sl];x=norm(pp);lx=x@L.T;rx=x@R.T;state=pp+(lx*rx)@D.T+bias;ts=torch.tanh(norm(state)@W.T/30);q=(30*ts).double().softmax(-1);teacher=(30*soft(p[sl])).double().softmax(-1);js=[]
  for v in directions:
   dxinp=((coord[sl]*T(v))@write.T).float();dx=dn(pp,dxinp);ds=dxinp+((dx@L.T)*rx+lx*(dx@R.T))@D.T;js.append(((dn(state,ds)@W.T)*(1-ts.square())).double())
  J=torch.stack(js,-1);mean=(teacher[:,:,None]*J).sum(1);center=J-mean[:,None,:];cov=torch.einsum('bv,bvi,bvj->bij',teacher,center,center);actual=mean-J[torch.arange(len(pp),device='cuda'),targets[sl]];kl=((q-teacher)[:,:,None]*J).sum(1)
  mean_residual=max(mean_residual,float((teacher[:,:,None]*center).sum(1).abs().max()));covs.append(cov.cpu().numpy());obs.append(actual.cpu().numpy());kls.append(kl.cpu().numpy())
  if start%256==240:print('variance contexts',start+16,flush=True)
 cov=np.concatenate(covs);obs=np.concatenate(obs);kl=np.concatenate(kls);refcontrast=(data['gradients'][:,2]-data['gradients'][:,1])@directions.T;refkl=data['gradients'][:,1]@directions.T;audits={'teacher_centering_max_absolute':mean_residual}
 for name,x,ref in [('contrast',obs,refcontrast),('KL',kl,refkl)]:
  err=float(np.linalg.norm(x-ref)/np.linalg.norm(ref));audits[name+'_gradient_relative_replay']=err;assert err<1e-4
 assert np.isfinite(cov).all() and np.isfinite(obs).all();assert np.linalg.eigvalsh(cov).min()>-1e-10;n=len(p);summary={}
 for i,name in enumerate(['KL_direction','contrast_direction']):
  vv=cov[:,i,i];refse=float(np.sqrt(vv.sum())/n);actualse=float(obs[:,i].std(ddof=1)/np.sqrt(n));summary[name]={'mean_observed_contrast':float(obs[:,i].mean()),'teacher_implied_mean_SE':refse,'empirical_mean_SE':actualse,'empirical_over_teacher_variance':float(actualse**2/refse**2),'observed_mean_over_teacher_SE':float(obs[:,i].mean()/refse),'observed_second_moment_over_teacher_variance':float(np.mean(obs[:,i]**2)/vv.mean()),'top1percent_contexts_variance_fraction':float(np.sort(vv)[-21:].sum()/vv.sum()),'top10percent_contexts_variance_fraction':float(np.sort(vv)[-205:].sum()/vv.sum())}
 blocks=[{'start':i,'mean':obs[i:i+512].mean(0).tolist(),'teacher_mean_SE':(np.sqrt(np.diagonal(cov[i:i+512].sum(0)))/512).tolist()} for i in range(0,n,512)]
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),conditional_covariance=cov,observed_contrast=obs,KL_projection=kl);out={'experiment':'H528','config':cfg,'summary':summary,'blocks':blocks,'teacher_mean_covariance':(cov.sum(0)/n**2).tolist(),'empirical_mean_covariance':(np.cov(obs,rowvar=False)/n).tolist(),'audits':audits,'unique_documents':26112};STEM.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()

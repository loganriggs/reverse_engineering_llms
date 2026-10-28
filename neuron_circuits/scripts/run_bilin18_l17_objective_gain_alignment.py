import json,hashlib,numpy as np,torch
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_objective_gain_alignment'
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 def T(x):return torch.tensor(x,device='cuda',dtype=torch.float64)
 z=np.load(ART/'bilin18_l17_source_sample_curve_models.npz');B=T(z['basis']);mu=T(z['mean']);C=T(z['C3072']);FM=T(z['F']);ev,V=torch.linalg.eigh(C);cs=(V*ev.sqrt())@V.T;ci=(V/ev.sqrt())@V.T;G=cs@FM@cs;_,U=torch.linalg.eigh((G+G.T)/2);u=U.flip(1)[:,:36];read=ci@u;write=B@cs@u
 A=cs@u@u.T@ci;replay=float((A-T(z['n3072_Fisher36'])).norm()/A.norm());assert replay<1e-8
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();eps=torch.finfo(torch.float32).eps
 def norm(x):return x/(x.square().mean(-1,keepdim=True)+eps).sqrt()
 def lp(p,double=False):
  ll,rr,dd,bb,ww=[x.double() for x in [L,R,D,bias,W]] if double else [L,R,D,bias,W]
  x=norm(p);st=p+((x@ll.T)*(x@rr.T))@dd.T+bb;return (30*torch.tanh(norm(st)@ww.T/30)).double().log_softmax(-1)
 def losses(g,p,h,coord,base,targets,double=False):
  err=h-(coord*g)@write.T;inp=p-err.to(p.dtype);q=lp(inp,double);raw=err.square().mean(-1);kl=(base.exp()*(base-q)).sum(-1);ce=(base-q).gather(1,targets[:,None]).squeeze(1);return torch.stack([raw,kl,ce],-1)
 panels={};saved={};grads={};baseline={};audits={'map_replay':replay};prior=np.load(ART/'bilin18_l17_source_sample_curve_rows.npz')
 for panel,file in cfg['evaluation_panels'].items():
  data=np.load(ART/file);p=torch.tensor(data['pre'],device='cuda');h=T(data['head_writes']).sum(1)-mu;coord=h@B@read;target=torch.tensor(data['targets'],device='cuda');panels[panel]=(p,h,coord,target);gg=[];vals=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);base=lp(p[sl]).detach();g=torch.ones((len(p[sl]),36),device='cuda',dtype=torch.float64,requires_grad=True);ls=losses(g,p[sl],h[sl],coord[sl],base,target[sl]);gs=[torch.autograd.grad(ls[:,i].sum(),g,retain_graph=i<2)[0].detach().cpu().numpy() for i in range(3)];gg.append(np.stack(gs,1));vals.append(ls.detach().cpu().numpy())
  grads[panel]=np.concatenate(gg);baseline[panel]=np.concatenate(vals);saved[panel+'_gradients']=grads[panel];saved[panel+'_baseline']=baseline[panel];ref=prior[panel+'_n3072_Fisher36'];re=float(abs(baseline[panel][:,1].mean()/ref[:,0].mean()-1));audits[panel+'_KL_endpoint_relative']=re;assert re<1e-4
  print('gradients',panel,grads[panel].mean(0).tolist(),flush=True)
 # Numerical audit on eight examples, double arithmetic with original normalization epsilon.
 p,h,coord,target=panels['H512'];p=p[:8].double();h=h[:8];coord=coord[:8];target=target[:8];base=lp(p,True).detach();g=torch.ones(36,device='cuda',dtype=torch.float64,requires_grad=True);ls=losses(g,p,h,coord,base,target,True).mean(0);direction=torch.arange(1,37,device='cuda',dtype=torch.float64);direction/=direction.norm();checks=[]
 for obj in range(3):
  ag=torch.autograd.grad(ls[obj],g,retain_graph=True)[0];pred=float(ag@direction)
  for step in [1e-3,5e-4]:
   with torch.no_grad():fd=float((losses(g+step*direction,p,h,coord,base,target,True).mean(0)[obj]-losses(g-step*direction,p,h,coord,base,target,True).mean(0)[obj])/(2*step))
   rel=abs(fd-pred)/max(abs(pred),1e-10);assert rel<1e-4;checks.append({'objective':obj,'step':step,'relative':rel})
 audits['double_finite_difference']=checks
 objectives=['activation_MSE','prediction_KL','observed_CE'];steps={}
 with torch.no_grad():
  for source,gg in grads.items():
   for oi,obj in enumerate(objectives):
    grad=gg[:,oi].mean(0);direction=-6*grad/np.linalg.norm(grad)
    for alpha in cfg['alphas']:
     gain=T(1+alpha*direction);key=f'{source}_{obj}_{alpha}';steps[key]={}
     for panel,(p,h,coord,target) in panels.items():
      vals=[]
      for start in range(0,512,16):
       sl=slice(start,start+16);vals.append(losses(gain,p[sl],h[sl],coord[sl],lp(p[sl]),target[sl]).cpu().numpy())
      rr=np.concatenate(vals);saved[key+'_'+panel]=rr;delta=rr-baseline[panel];steps[key][panel]={'mean_change':delta.mean(0).tolist(),'paired_SE':(delta.std(0,ddof=1)/np.sqrt(512)).tolist(),'linear_prediction':(grads[panel].mean(0)@(alpha*direction)).tolist()}
 rng=np.random.default_rng(cfg['bootstrap_seed']);means={p:g.mean(0) for p,g in grads.items()};boots={}
 for p,g in grads.items():
  weights=rng.multinomial(512,np.ones(512)/512,size=cfg['bootstrap_draws'])/512;boots[p]=(weights@g.reshape(512,-1)).reshape(-1,3,36)
 def cos(x,y):return (x*y).sum(-1)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1))
 alignment={}
 for panel in panels:
  for i in range(3):
   for k in range(i+1,3):
    boot=cos(boots[panel][:,i],boots[panel][:,k]);alignment[panel+'_'+objectives[i]+'__'+objectives[k]]={'cosine':float(cos(means[panel][i],means[panel][k])),'bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'coordinate_sign_agreement':float(np.mean(np.sign(means[panel][i])==np.sign(means[panel][k])))}
 for i,obj in enumerate(objectives):
  boot=cos(boots['H512'][:,i],boots['H521'][:,i]);alignment['cross_panel_'+obj]={'cosine':float(cos(means['H512'][i],means['H521'][i])),'bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved,read=read.cpu().numpy(),write=write.cpu().numpy());STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H525','config':cfg,'objectives':objectives,'alignment':alignment,'steps':steps,'audits':audits,'baseline_means':{p:v.mean(0).tolist() for p,v in baseline.items()},'unique_documents':24064},indent=2)+'\n');print('alignment',alignment,flush=True)
if __name__=='__main__':main()

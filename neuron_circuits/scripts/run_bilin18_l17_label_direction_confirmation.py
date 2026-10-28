import json,hashlib,numpy as np,torch
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_label_direction_confirmation'
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
 docs=np.load(ROOT/cfg['corpus']);p,heads,split=collect_sources(model,docs,cfg);assert split<1e-5
 h=heads.double().sum(1)-mu;coord=h@B@read;target=torch.tensor(docs[:,64].astype(np.int64),device='cuda');gg=[];vals=[]
 for start in range(0,cfg['rows'],16):
  sl=slice(start,start+16);base=lp(p[sl]).detach();g=torch.ones((len(p[sl]),36),device='cuda',dtype=torch.float64,requires_grad=True);ls=losses(g,p[sl],h[sl],coord[sl],base,target[sl]);gs=[torch.autograd.grad(ls[:,i].sum(),g,retain_graph=i<2)[0].detach().cpu().numpy() for i in range(3)];gg.append(np.stack(gs,1));vals.append(ls.detach().cpu().numpy())
  if start%256==240:print('gradients',start+16,flush=True)
 gradients=np.concatenate(gg);values=np.concatenate(vals);directions=np.load(ART/'bilin18_l17_label_direction_confirmation_directions.npz')['directions'];projection=np.stack([gradients[:,1]@directions[0],(gradients[:,2]-gradients[:,1])@directions[1]],1);n=len(projection);est=projection.mean(0);se=projection.std(0,ddof=1)/np.sqrt(n);assert (se>0).all()
 rng=np.random.default_rng(cfg['bootstrap_seed']);idx=rng.integers(0,n,(cfg['bootstrap_draws'],n));draw=projection[idx].mean(1);q=float(np.quantile(np.max(abs((draw-est)/se),axis=1),.95));joint={'names':['KL','CE_minus_KL'],'mean_projection':est.tolist(),'SE':se.tolist(),'lower':(est-q*se).tolist(),'upper':(est+q*se).tolist(),'positive_lower':(est-q*se>0).tolist(),'critical_value':q}
 blocks=[{'start':i,'mean_projection':projection[i:i+512].mean(0).tolist(),'SE':(projection[i:i+512].std(0,ddof=1)/np.sqrt(512)).tolist()} for i in range(0,n,512)]
 assert np.isfinite(gradients).all() and np.isfinite(values).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),gradients=gradients,losses=values,projections=projection,pre=p.detach().cpu().numpy(),head_writes=heads.detach().cpu().numpy(),targets=target.cpu().numpy());out={'experiment':'H527','config':cfg,'joint_projections':joint,'blocks':blocks,'baseline_means':values.mean(0).tolist(),'audits':{'map_replay':replay,'head_sum_relative':split},'unique_documents':26112};STEM.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()

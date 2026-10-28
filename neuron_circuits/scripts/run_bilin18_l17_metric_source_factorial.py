import json, hashlib, numpy as np, torch
from run_bilin18_l17_local_residual_charts import ROOT, ART, load_full_model
STEM=ART/'bilin18_l17_metric_source_factorial'
@torch.inference_mode()
def main():
 torch.set_num_threads(6)
 cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items(): assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 z=np.load(ART/'bilin18_l17_joint_fisher_rank_curve_model.npz'); c=np.load(ART/'bilin18_l17_joint_pca_more_calibration_models.npz'); v=np.load(ART/'bilin18_l17_joint_source_confirmation_rows.npz')
 def T(x): return torch.tensor(x,device='cuda',dtype=torch.float64)
 B,mu,cs,U,Ftrain=T(z['basis']),T(z['mean']),T(z['covariance']).sqrt(),T(z['U']),T(z['F'])
 errors={}
 for name,h in [('cal_shared_mean',T(c['calibration_sum'])-mu),('eval_shared_mean',T(v['head_writes']).sum(1)-mu)]:
  hc=h@B; ee={'mean':hc}
  for k in cfg['ranks']:
   e=hc.clone();e[:,:k]=0;ee[f'PCA{k}']=e;u=U[:,:k];ee[f'Fisher{k}']=hc-((hc/cs)@u)@u.T*cs
  errors[name]=ee
 names=list(errors['eval_shared_mean']);E=torch.stack([errors['eval_shared_mean'][n] for n in names],1)
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 L,R,D=[m.weight.detach().double() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach().double();W=model.lm_head.weight.detach();Wd=W.double();eps=torch.finfo(torch.float32).eps
 def logits(t):
  x=t/torch.sqrt(t.square().mean()+eps);st=t+D@((L@x)*(R@x))+bias;return 30*torch.tanh((Wd@(st/torch.sqrt(st.square().mean()+eps)))/30)
 FM=torch.zeros_like(Ftrain);natural=[];audits={}
 for i,t in enumerate(T(v['pre'])):
  scale=torch.sqrt(t.square().mean()+eps);x=t/scale;dx=B/scale-t[:,None]*(t@B)[None,:]/(1152*scale**3)
  dy=B+D@((R@x)[:,None]*(L@dx)+(L@x)[:,None]*(R@dx));st=t+D@((L@x)*(R@x))+bias;sc=torch.sqrt(st.square().mean()+eps);dn=dy/sc-st[:,None]*(st@dy)[None,:]/(1152*sc**3)
  soft=torch.tanh((Wd@(st/sc))/30);prob=(30*soft).softmax(-1);J=(W@dn.float()).double()*(1-soft.square())[:,None];J-=(prob[:,None]*J).sum(0);weighted=(prob.sqrt()[:,None]*J).float();Fi=(weighted.T@weighted).double();FM+=Fi/len(v['pre']);natural.append((.5*((E[i]@Fi)*E[i]).sum(-1)).cpu().numpy())
  if i==0:
   analytic=(Wd@dn[:,0])*(1-soft.square());step=float(t.norm())*1e-4;errs=[]
   for ss in [step,step/2]:errs.append(float(((logits(t+ss*B[:,0])-logits(t-ss*B[:,0]))/(2*ss)-analytic).norm()/analytic.norm()))
   assert max(errs)<1e-4;audits['derivative_relative_errors']=errs
  if i%64==63:print('evaluation metric receivers',i+1,flush=True)
 audits['metric_symmetry_max']=float((FM-FM.T).abs().max());FM=(FM+FM.T)/2
 eig=torch.linalg.eigvalsh(FM);audits['metric_min_eigenvalue']=float(eig.min());assert eig.min()>-1e-6*eig.max()
 natural=np.array(natural);assert np.isfinite(natural).all() and natural.min()>-1e-9
 metrics={};saved={'F_eval':FM.cpu().numpy(),'natural_local_KL':natural};prior=json.loads((ART/'bilin18_l17_source_moment_diagnostic.json').read_text());replay=0.
 for metric,F in [('train64',Ftrain),('eval512',FM)]:
  metrics[metric]={}
  for panel,es in errors.items():
   mm={}
   for n,e in es.items():
    rows=(.5*((e@F)*e).sum(-1)).cpu().numpy();assert np.isfinite(rows).all() and rows.min()>-1e-9;saved[metric+'_'+panel+'_'+n]=rows;mm[n]={'mean_local_KL':float(rows.mean())}
    if metric=='train64':replay=max(replay,abs(rows.mean()/prior['metrics'][panel][n]['mean_local_KL']-1))
   for n in names:mm[n]['recovery']=1-mm[n]['mean_local_KL']/mm['mean']['mean_local_KL']
   metrics[metric][panel]=mm
 assert replay<1e-9;audits['H518_max_relative_replay_error']=replay
 old=np.load(ART/'bilin18_l17_source_context_pairing_rows.npz')['local_KL'][0];assert old.shape==natural.shape
 nm={n:{'mean_local_KL':float(natural[:,i].mean()),'recovery':float(1-natural[:,i].mean()/natural[:,0].mean()),'full_vector_H517_local_KL':float(old[:,i].mean()),'supported_over_full_error':float(natural[:,i].mean()/old[:,i].mean())} for i,n in enumerate(names)}
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved)
 out={'experiment':'H519','config':cfg,'metrics':metrics,'natural':nm,'names':names,'audits':audits,'unique_documents':21504}
 STEM.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'natural':nm,'audits':audits}),flush=True)
if __name__=='__main__':main()

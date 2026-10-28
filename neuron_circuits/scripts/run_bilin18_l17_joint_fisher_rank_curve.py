import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_joint_fisher_rank_curve'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();z=np.load(ART/'bilin18_l17_joint_pca_more_calibration_models.npz');B=torch.tensor(z['pooled1024_basis'],device='cuda');rank=B.shape[1];assert rank==1023;cs=torch.tensor(z['pooled1024_singular_values'][:rank],device='cuda')/np.sqrt(1023);mu=torch.tensor(z['shared_reconstruction_mean'],device='cuda');old=np.load(ART/'bilin18_l17_joint_source_confirmation_rows.npz');p=torch.tensor(old['pre'],device='cuda');h=torch.tensor(old['head_writes'],device='cuda').double().sum(1)-mu;targets=torch.tensor(old['targets'],device='cuda');hc=h@B;L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();Ld,Rd,Dd,bd,Wd=L.double(),R.double(),D.double(),bias.double(),W.double();train=torch.tensor(np.load(ART/'bilin18_l17_canonical_learning_curve_rows.npz')['pre'][cfg['metric_receiver_ids']],device='cuda').double();eps=torch.finfo(torch.float32).eps;FM=torch.zeros((rank,rank),device='cuda',dtype=torch.float64);audits={'basis_orthogonality':float((B.T@B-torch.eye(rank,device='cuda')).abs().max())};assert audits['basis_orthogonality']<1e-10
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 def full_logits(inp):
  x=inp/torch.sqrt(inp.square().mean()+eps);st=inp+Dd@((Ld@x)*(Rd@x))+bd;return 30*torch.tanh(Wd@(st/torch.sqrt(st.square().mean()+eps))/30)
 for ti,t in enumerate(train):
  scale=torch.sqrt(t.square().mean()+eps);x=t/scale;dx=B/scale-t[:,None]*(t@B)[None,:]/(1152*scale**3);dy=B+Dd@((Rd@x)[:,None]*(Ld@dx)+(Ld@x)[:,None]*(Rd@dx));st=t+Dd@((Ld@x)*(Rd@x))+bd;sc=torch.sqrt(st.square().mean()+eps);dn=dy/sc-st[:,None]*(st@dy)[None,:]/(1152*sc**3);logits=Wd@(st/sc);tanh=torch.tanh(logits/30);prob=(30*tanh).softmax(-1);J=(W@dn.float()).double()*(1-tanh.square())[:,None];J-=(prob[:,None]*J).sum(0);weighted=(prob.sqrt()[:,None]*J).float();FM+=(weighted.T@weighted).double()/len(train)
  if ti==0:
   analytic=(Wd@dn[:,0])*(1-tanh.square());step=float(t.norm())*1e-4;errs=[]
   for ss in [step,step/2]:
    num=(full_logits(t+ss*B[:,0])-full_logits(t-ss*B[:,0]))/(2*ss);errs.append(float((num-analytic).norm()/analytic.norm()))
   assert max(errs)<1e-4;audits['derivative_relative_errors']=errs
  if ti%16==15:print('metric receivers',ti+1,flush=True)
 FM=(FM+FM.T)/2;G=cs[:,None]*FM*cs[None,:];ev,U=torch.linalg.eigh(G);ev=ev.flip(0);U=U.flip(1);assert float(ev.min())>-1e-6*float(ev.max());np.savez_compressed(STEM.with_name(STEM.name+'_model.npz'),basis=B.cpu().numpy(),mean=mu.cpu().numpy(),covariance=cs.square().cpu().numpy(),F=FM.cpu().numpy(),U=U.cpu().numpy(),eigenvalues=ev.cpu().numpy());print('MODEL SAVED BEFORE EVALUATION',flush=True);saved={};metrics={};geometry={};y0=native(p);pca=np.load(ART/'bilin18_l17_joint_pca_rank_curve_rows.npz');den=old['mean'][:,0].mean();eye=torch.eye(rank,device='cuda',dtype=torch.float64)
 for k in cfg['ranks']:
  u=U[:,:k];A=cs[:,None]*(u@u.T)/cs[None,:];hat=hc@A.T@B.T;error=h-hat;res=(eye-A)*cs[None,:];weighted_loss=torch.trace(FM@res@res.T);pc=np.trace(G[k:,k:].cpu().numpy());assert float(weighted_loss)-pc<1e-7*float(torch.trace(G));geometry[str(k)]={'operator_norm':float(torch.linalg.matrix_norm(A,ord=2)),'training_Fisher_recovery':float(1-weighted_loss/torch.trace(G)),'training_PCA_recovery':float(1-pc/float(torch.trace(G))),'idempotence_relative':float((A@A-A).norm()/A.norm())};assert geometry[str(k)]['idempotence_relative']<1e-8;inp=p-error.float();yp=native(inp);rows=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);base=lp(p[sl]+y0[sl]);q=lp(inp[sl]+yp[sl]);rows.append(torch.stack([(base.exp()*(base-q)).sum(-1),(base-q).gather(1,targets[sl,None]).squeeze(1)],-1).cpu().numpy())
  rr=np.concatenate(rows);saved[f'rank{k}']=rr;pr=pca[f'rank{k}'];metrics[str(k)]={'mean_KL':float(rr[:,0].mean()),'KL_recovery_vs_mean':float(1-rr[:,0].mean()/den),'PCA_KL':float(pr[:,0].mean()),'PCA_recovery':float(1-pr[:,0].mean()/den),'gain_recovery':float((pr[:,0].mean()-rr[:,0].mean())/den),'mean_CE_change':float(rr[:,1].mean()),'CE_change_SE':float(rr[:,1].std(ddof=1)/np.sqrt(512)),'source_raw_retention':float(1-error.square().sum()/h.square().sum())}
  if k==1023:
   audits['full_span_reconstruction_relative']=float((hat-hc@B.T).norm()/h.norm());assert audits['full_span_reconstruction_relative']<1e-8;audits['full_rank_KL_replay_relative']=float(abs(rr[:,0].sum()-pr[:,0].sum())/pr[:,0].sum());assert audits['full_rank_KL_replay_relative']<1e-4
  print(k,metrics[str(k)],geometry[str(k)],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H516','config':cfg,'metrics':metrics,'geometry':geometry,'audits':audits,'covariance_condition_number':float((cs.max()/cs.min())**2),'mean_replacement_KL':den,'unique_documents':21504},indent=2)+'\n')
if __name__=='__main__':main()

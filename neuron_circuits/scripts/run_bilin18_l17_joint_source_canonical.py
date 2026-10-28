import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_joint_source_canonical'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();_,calheads,split=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),cfg);ids=np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['calibration_ids'];cal=calheads[ids].double().sum(1);mu=cal.mean(0);center=cal-mu;_,sv,Vh=torch.linalg.svd(center,full_matrices=False);rank=int((sv>sv[0]*cfg['relative_singular_cutoff']).sum());assert rank>=36;B=Vh[:rank].T;cs=sv[:rank]/np.sqrt(len(ids)-1);old=np.load(ART/'bilin18_l17_allocated_source_confirmation_rows.npz');p=torch.tensor(old['pre'],device='cuda');h=torch.tensor(old['head_writes'],device='cuda').double().sum(1)-mu;targets=torch.tensor(old['targets'],device='cuda');hc=h@B;L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();Ld,Rd,Dd,bd,Wd=L.double(),R.double(),D.double(),bias.double(),W.double();train=torch.tensor(np.load(ART/'bilin18_l17_canonical_learning_curve_rows.npz')['pre'][cfg['metric_receiver_ids']],device='cuda').double();eps=torch.finfo(torch.float32).eps;FM=torch.zeros((rank,rank),device='cuda',dtype=torch.float64);audits={'head_sum_relative':split}
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
 FM=(FM+FM.T)/2;G=cs[:,None]*FM*cs[None,:];ev,U=torch.linalg.eigh(G);U=U.flip(1)[:,:36];A=cs[:,None]*(U@U.T)/cs[None,:];maps={'mean':torch.zeros_like(h),'joint_PCA':hc[:,:36]@B[:,:36].T,'joint_Fisher':hc@A.T@B.T,'empirical_span':hc@B.T};identity=torch.eye(rank,device='cuda',dtype=torch.float64);res=(identity-A)*cs[None,:];weighted_loss=torch.trace(FM@res@res.T);pcaloss=torch.trace(G[36:,36:]);assert float(weighted_loss-pcaloss)<1e-8*float(torch.trace(G));calerr=float((center-(center@B)@B.T).norm()/center.norm());assert calerr<1e-8;audits['calibration_span_relative']=calerr;geometry={'calibration_rows':len(ids),'empirical_rank':rank,'heldout_outside_span_energy_fraction':float((h-hc@B.T).square().sum()/h.square().sum()),'calibration_top36_energy_fraction':float(sv[:36].square().sum()/sv.square().sum()),'Fisher_training_recovery':float(1-weighted_loss/torch.trace(G)),'PCA_training_recovery':float(1-pcaloss/torch.trace(G)),'covariance_condition_number':float((cs.max()/cs.min())**2)};np.savez_compressed(STEM.with_name(STEM.name+'_model.npz'),basis=B.cpu().numpy(),mean=mu.cpu().numpy(),covariance=cs.square().cpu().numpy(),F=FM.cpu().numpy(),A=A.cpu().numpy(),U=U.cpu().numpy(),singular_values=sv.cpu().numpy());saved={};metrics={};y0=native(p)
 for name,hhat in maps.items():
  error=h-hhat;inp=p-error.float();yp=native(inp);rows=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);base=lp(p[sl]+y0[sl]);q=lp(inp[sl]+yp[sl]);rows.append(torch.stack([(base.exp()*(base-q)).sum(-1),(base-q).gather(1,targets[sl,None]).squeeze(1)],-1).cpu().numpy())
  rr=np.concatenate(rows);saved[name]=rr;metrics[name]={'mean_KL':float(rr[:,0].mean()),'mean_CE_change':float(rr[:,1].mean()),'CE_change_SE':float(rr[:,1].std(ddof=1)/np.sqrt(512)),'source_raw_recovery':float(1-error.square().sum()/h.square().sum())}
  if name=='mean':
   ref=old['full_pre_MLP_mean'][:,:2];drift=float(np.max(np.abs(rr-ref)));assert drift<1e-7;audits['mean_replay_max_absolute']=drift
  print(name,metrics[name],flush=True)
 for name in metrics:metrics[name]['KL_recovery_vs_mean']=float(1-metrics[name]['mean_KL']/metrics['mean']['mean_KL'])
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H511','config':cfg,'metrics':metrics,'geometry':geometry,'audits':audits,'allocated_reference':json.loads((ART/'bilin18_l17_allocated_source_confirmation.json').read_text())['metrics']['full_pre_MLP']['allocated'],'unique_documents':19968},indent=2)+'\n')
if __name__=='__main__':main()

import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_full_path_metric'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();data=np.load(ART/'bilin18_l17_natural_source_replacement_rows.npz');p=torch.tensor(data['pre'],device='cuda');dh=torch.tensor(data['head_writes'],device='cuda').double();targets=torch.tensor(data['targets'],device='cuda');compiled=np.load(ART/'bilin18_l17_fisher_source_rank_compiled.npz');pooled=np.load(ART/'bilin18_l17_fisher_pooled_metric_models.npz');L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();saved={};metrics={};errs={n:torch.zeros_like(p,dtype=torch.float64) for n in cfg['methods']};total=torch.zeros_like(p,dtype=torch.float64)
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p);Ld,Rd,Dd,bd,Wd=L.double(),R.double(),D.double(),bias.double(),W.double();eps=torch.finfo(torch.float32).eps;train=torch.tensor(np.load(ART/'bilin18_l17_canonical_learning_curve_rows.npz')['pre'][cfg['metric_receiver_ids']],device='cuda').double();models={};audits={};geometry={};old=np.load(ART/'bilin18_l17_joint_source_paths_rows.npz')
 def full_logits(inp):
  x=inp/torch.sqrt(inp.square().mean()+eps);st=inp+Dd@((Ld@x)*(Rd@x))+bd;return 30*torch.tanh(Wd@(st/torch.sqrt(st.square().mean()+eps))/30)
 def fit_full(B,cs,head):
  FM=torch.zeros((128,128),device='cuda',dtype=torch.float64)
  for ti,t in enumerate(train):
   scale=torch.sqrt(t.square().mean()+eps);x=t/scale;dx=B/scale-t[:,None]*(t@B)[None,:]/(1152*scale**3);dy=B+Dd@((Rd@x)[:,None]*(Ld@dx)+(Ld@x)[:,None]*(Rd@dx));st=t+Dd@((Ld@x)*(Rd@x))+bd;sc=torch.sqrt(st.square().mean()+eps);dn=dy/sc-st[:,None]*(st@dy)[None,:]/(1152*sc**3);logits=Wd@(st/sc);tanh=torch.tanh(logits/30);prob=(30*tanh).softmax(-1);J=(W@dn.float()).double()*(1-tanh.square())[:,None];J-=(prob[:,None]*J).sum(0);weighted=(prob.sqrt()[:,None]*J).float();FM+=(weighted.T@weighted).double()/len(train)
   if ti==0:
    analytic=(Wd@dn[:,0])*(1-tanh.square());step=float(t.norm())*1e-4;errors=[]
    for ss in [step,step/2]:
     num=(full_logits(t+ss*B[:,0])-full_logits(t-ss*B[:,0]))/(2*ss);errors.append(float((num-analytic).norm()/analytic.norm()))
    assert max(errors)<1e-4;audits[str(head)]={'full_derivative_relative_errors':errors}
  FM=(FM+FM.T)/2;G=cs[:,None]*FM*cs[None,:];ev,U=torch.linalg.eigh(G);U=U.flip(1)[:,:4];A=cs[:,None]*(U@U.T)/cs[None,:];reference=torch.tensor(pooled[f'h{head}_A'],device='cuda');identity=torch.eye(128,device='cuda',dtype=torch.float64)
  def loss(AA):
   residual=(identity-AA)*cs[None,:];return torch.trace(FM@residual@residual.T)
  new=loss(A);ref=loss(reference);assert float(new-ref)<1e-8*float(torch.trace(G));geometry[str(head)]={'full_metric_objective_recovery':float(1-new/torch.trace(G)),'MLP_metric_objective_recovery':float(1-ref/torch.trace(G))};models[f'h{head}_F']=FM.cpu().numpy();models[f'h{head}_A']=A.cpu().numpy();models[f'h{head}_U']=U.cpu().numpy();print('full metric fitted',head,flush=True);return A

 for head in cfg['heads']:
  B=torch.tensor(compiled[f'h{head}_basis'],device='cuda');cs=torch.tensor(compiled[f'h{head}_covariance'],device='cuda').sqrt();U=torch.tensor(compiled[f'h{head}_weighted_eigenvectors'][:,:4],device='cuda');A=torch.tensor(pooled[f'h{head}_A'],device='cuda');fullA=fit_full(B,cs,head);h=dh[:,head]-torch.tensor(compiled[f'h{head}_mean'],device='cuda');Q=torch.linalg.qr(B@(cs[:,None]*U)).Q;hc=h@B;maps={'mean':torch.zeros_like(h),'PCA':hc[:,:4]@B[:,:4].T,'orthogonal':(h@Q)@Q.T,'original':(((hc/cs)@U)@U.T*cs)@B.T,'pooled':hc@A.T@B.T,'full_metric':hc@fullA.T@B.T};total+=h
  for n in errs:errs[n]+=h-maps[n]
 np.savez_compressed(STEM.with_name(STEM.name+'_models.npz'),**models)
 for path in cfg['paths']:
  metrics[path]={}
  for name,error in errs.items():
   inp=p-error.float();yp=native(inp);bypass=p if path=='MLP_only' else inp;rows=[]
   for start in range(0,512,16):
    sl=slice(start,start+16);base=lp(p[sl]+y0[sl]);q=lp(bypass[sl]+yp[sl]);rows.append(torch.stack([(base.exp()*(base-q)).sum(-1),(base-q).gather(1,targets[sl,None]).squeeze(1),(bypass[sl].double()+yp[sl].double()-p[sl].double()-y0[sl].double()).square().sum(-1)],-1).cpu().numpy())
   rr=np.concatenate(rows);
   if name!='full_metric':assert np.allclose(rr,old[f'{path}_{name}'],rtol=1e-5,atol=1e-9)
   saved[f'{path}_{name}']=rr;metrics[path][name]={'mean_KL':float(rr[:,0].mean()),'mean_CE_change':float(rr[:,1].mean()),'CE_change_SE':float(rr[:,1].std(ddof=1)/np.sqrt(512)),'mean_state_squared_error':float(rr[:,2].mean()),'source_raw_recovery':float(1-error.square().sum()/total.square().sum())}
  for name in errs:metrics[path][name]['KL_recovery_vs_mean']=float(1-metrics[path][name]['mean_KL']/metrics[path]['mean']['mean_KL'])
  print(path,metrics[path],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H507','config':cfg,'metrics':metrics,'audits':audits,'geometry':geometry,'columns':['KL_original_to_replacement','observed_token_CE_change','final_state_squared_error'],'unique_documents':19456},indent=2)+'\n')
if __name__=='__main__':main()

import json,hashlib,itertools,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_fisher_metric_stability'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();old=np.load(ART/'bilin18_l17_fisher_source_donor_transfer_rows.npz');p=torch.tensor(old['pre'],device='cuda');sources=np.load(ART/'bilin18_l17_fisher_source_donor_transfer_sources.npz');compiled=np.load(ART/'bilin18_l17_fisher_source_rank_compiled.npz');L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();Ld,Rd,Dd,bd,Wd=L.double(),R.double(),D.double(),bias.double(),W.double();train=torch.tensor(np.load(ART/'bilin18_l17_canonical_learning_curve_rows.npz')['pre'][:1024],device='cuda').double();saved={};models={};metrics={};geometry={};audits={};eps=torch.finfo(torch.float32).eps;pairs=list(itertools.combinations(range(4),2))
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p)
 for head in cfg['heads']:
  B=torch.tensor(compiled[f'h{head}_basis'],device='cuda');cs=torch.tensor(compiled[f'h{head}_covariance'],device='cuda').sqrt();h=torch.tensor(sources[f'h{head}_sources'],device='cuda');hc=h@B;As=[];Qs=[];preds=[];geometry[str(head)]={'groups':{},'pairs':{}};audits[str(head)]={};metrics[str(head)]={}
  for gi,ids in enumerate(cfg['metric_groups']):
   FM=torch.zeros((128,128),device='cuda',dtype=torch.float64)
   for t in train[ids]:
    scale=torch.sqrt(t.square().mean()+eps);x=t/scale;dx=B/scale-t[:,None]*(t@B)[None,:]/(1152*scale**3);dy=Dd@((Rd@x)[:,None]*(Ld@dx)+(Ld@x)[:,None]*(Rd@dx));st=t+Dd@((Ld@x)*(Rd@x))+bd;sc=torch.sqrt(st.square().mean()+eps);dn=dy/sc-st[:,None]*(st@dy)[None,:]/(1152*sc**3);logits=Wd@(st/sc);tanh=torch.tanh(logits/30);prob=(30*tanh).softmax(-1);J=(W@dn.float()).double()*(1-tanh.square())[:,None];J-=(prob[:,None]*J).sum(0);weighted=(prob.sqrt()[:,None]*J).float();FM+=(weighted.T@weighted).double()/len(ids)
   FM=(FM+FM.T)/2;G=cs[:,None]*FM*cs[None,:];ev,U=torch.linalg.eigh(G);ev=ev.flip(0);U=U.flip(1);u=U[:,:4];A=cs[:,None]*(u@u.T)/cs[None,:];As.append(A);Qs.append(torch.linalg.qr(cs[:,None]*u).Q);preds.append(((hc@A.T)@B.T)[np.arange(512)%64].float());models[f'h{head}_g{gi}_A']=A.cpu().numpy();models[f'h{head}_g{gi}_F']=FM.cpu().numpy();models[f'h{head}_g{gi}_U']=u.cpu().numpy();geometry[str(head)]['groups'][str(gi)]={'top4_weighted_energy':float(ev[:4].sum()/ev.sum()),'relative_gap4_5':float((ev[3]-ev[4])/ev[3]),'operator_norm':float(torch.linalg.matrix_norm(A,ord=2))}
   if gi==0:
    ref=torch.tensor(compiled[f'h{head}_fisher'],device='cuda');err=float((FM-ref).norm()/ref.norm());assert err<1e-6;audits[str(head)]['reference_metric_relative']=err
   print('metric',head,gi,flush=True)
  for i,k in pairs:
   cos=torch.linalg.svdvals(Qs[i].T@Qs[k]).clamp(0,1);aa=As[i]*cs[None,:];bb=As[k]*cs[None,:];geometry[str(head)]['pairs'][f'{i}_{k}']={'write_mean_cosine_squared':float(cos.square().mean()),'write_max_angle_degrees':float(torch.rad2deg(torch.acos(cos.min()))),'covariance_weighted_operator_relative_difference':float((aa-bb).norm()/torch.sqrt((aa.square().sum()+bb.square().sum())/2))}
  hh=h[np.arange(512)%64].float()
  for sign,alpha in [('negative',-1),('positive',1)]:
   yf=native(p-alpha*hh);ys=[native(p-alpha*v) for v in preds];rows=[]
   for start in range(0,512,16):
    sl=slice(start,start+16);tf=lp(p[sl]+yf[sl]);pf=tf.exp();ps=[lp(p[sl]+v[sl]) for v in ys];cols=[(pf*(tf-lp(p[sl]+y0[sl]))).sum(-1)];cols.extend((pf*(tf-pp)).sum(-1) for pp in ps);cols.extend(.5*((ps[i].exp()*(ps[i]-ps[k])).sum(-1)+(ps[k].exp()*(ps[k]-ps[i])).sum(-1)) for i,k in pairs);rows.append(torch.stack(cols,-1).cpu().numpy())
   rr=np.concatenate(rows);saved[f'h{head}_{sign}']=rr;v=rr.sum(0);prev=old[f'h{head}_{sign}'];drift=float(abs(v[1]-prev[:,2].sum())/v[0]);assert drift<1e-5;audits[str(head)][sign+'_reference_recovery_drift']=drift;metrics[str(head)][sign]={'KL_recovery':[float(1-v[i+1]/v[0]) for i in range(4)],'pair_symmetric_KL_over_native_effect':{f'{i}_{k}':float(v[5+pi]/v[0]) for pi,(i,k) in enumerate(pairs)}}
  print('head',head,'metrics',metrics[str(head)],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);np.savez_compressed(STEM.with_name(STEM.name+'_models.npz'),**models);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H503','config':cfg,'metrics':metrics,'geometry':geometry,'audits':audits,'columns':['effect_KL']+[f'group{i}_error_KL' for i in range(4)]+[f'pair{i}_{k}_symmetric_KL' for i,k in pairs],'unique_documents':19456},indent=2)+'\n')
if __name__=='__main__':main()

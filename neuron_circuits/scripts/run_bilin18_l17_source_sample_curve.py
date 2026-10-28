import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_source_sample_curve'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 def T(x):return torch.tensor(x,device='cuda',dtype=torch.float64)
 z=np.load(ART/'bilin18_l17_metric_sample_curve_models.npz');B=T(z['basis']);mu=T(z['mean']);FM=T(z['F1024']);old=np.load(ART/'bilin18_l17_joint_pca_more_calibration_models.npz')
 precal,heads,split=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);assert split<1e-5;new=heads.double().sum(1);cal=torch.cat([T(old['calibration_sum']),new]);np.savez_compressed(STEM.with_name(STEM.name+'_sources.npz'),new_sum=new.cpu().numpy(),new_pre=precal.cpu().numpy())
 maps={};compiled={'basis':B.cpu().numpy(),'mean':mu.cpu().numpy(),'F':FM.cpu().numpy()};audits={'head_sum_relative':split};geometry={};eye=torch.eye(B.shape[1],device='cuda',dtype=torch.float64)
 for n in cfg['sample_sizes']:
  xx=cal[:n];xc=(xx-xx.mean(0))@B;C=xc.T@xc/(n-1);C=(C+C.T)/2;ev,V=torch.linalg.eigh(C);assert ev.min()>0;cs=(V*ev.sqrt())@V.T;ci=(V/ev.sqrt())@V.T;G=cs@FM@cs;G=(G+G.T)/2;fv,U=torch.linalg.eigh(G);U=U.flip(1);P=V.flip(1);geometry[str(n)]={'covariance_condition':float(ev.max()/ev.min()),'root_inverse_error':float((cs@ci-eye).abs().max())};assert geometry[str(n)]['root_inverse_error']<1e-7
  compiled[f'C{n}']=C.cpu().numpy()
  for k in cfg['ranks']:
   u=U[:,:k];p=P[:,:k];aa={'PCA':p@p.T,'Fisher':cs@u@u.T@ci}
   for method,A in aa.items():
    name=f'n{n}_{method}{k}';maps[name]=A;compiled[name]=A.cpu().numpy();err=float((A@A-A).norm()/A.norm());assert err<1e-7
    if n==1024:
     if method=='PCA':ref=eye.clone();ref[k:]=0
     else:
      uu=T(z['U1024'][:,:k]);ss=T(z['covariance']).sqrt();ref=ss[:,None]*(uu@uu.T)/ss[None,:]
     replay=float((A-ref).norm()/ref.norm());audits[name+'_map_replay']=replay;assert replay<1e-6
  print('fit source samples',n,geometry[str(n)],flush=True)
 np.savez_compressed(STEM.with_name(STEM.name+'_models.npz'),**compiled);print('ALL MAPS SAVED BEFORE EVALUATION',flush=True)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach()
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 saved={};metrics={}
 for panel,filename in cfg['evaluation_panels'].items():
  data=np.load(ART/filename);p=torch.tensor(data['pre'],device='cuda');h=T(data['head_writes']).sum(1)-mu;hc=h@B;targets=torch.tensor(data['targets'],device='cuda');y=native(p);den=data['mean'][:,0].mean();metrics[panel]={}
  for name,A in maps.items():
   error=h-hc@A.T@B.T;inp=p-error.float();yp=native(inp);rows=[]
   for start in range(0,len(p),16):
    sl=slice(start,start+16);base=lp(p[sl]+y[sl]);q=lp(inp[sl]+yp[sl]);rows.append(torch.stack([(base.exp()*(base-q)).sum(-1),(base-q).gather(1,targets[sl,None]).squeeze(1)],-1).cpu().numpy())
   rr=np.concatenate(rows);assert np.isfinite(rr).all();saved[panel+'_'+name]=rr;metrics[panel][name]={'mean_KL':float(rr[:,0].mean()),'recovery':float(1-rr[:,0].mean()/den),'mean_CE_change':float(rr[:,1].mean()),'CE_change_SE':float(rr[:,1].std(ddof=1)/np.sqrt(len(p)))}
   if name.startswith('n1024_') and panel=='H521':
    suffix=name.split('_')[1];ref=data['n1024_rank'+suffix[6:]] if suffix.startswith('Fisher') else data[suffix];er=abs(rr[:,0].mean()/ref[:,0].mean()-1);audits[name+'_finite_KL_replay']=float(er);assert er<1e-4
  print(panel,metrics[panel],flush=True)
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H522','config':cfg,'metrics':metrics,'geometry':geometry,'audits':audits,'unique_documents':24064},indent=2)+'\n')
if __name__=='__main__':main()

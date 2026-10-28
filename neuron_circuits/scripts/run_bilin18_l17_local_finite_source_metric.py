import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_local_finite_source_metric'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();old=np.load(ART/'bilin18_l17_fisher_source_donor_transfer_rows.npz');oldortho=np.load(ART/'bilin18_l17_fisher_orthogonal_control_rows.npz');p=torch.tensor(old['pre'],device='cuda');sources=np.load(ART/'bilin18_l17_fisher_source_donor_transfer_sources.npz');compiled=np.load(ART/'bilin18_l17_fisher_source_rank_compiled.npz');L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();saved={};metrics={};audits={};eps=torch.finfo(torch.float32).eps;labels=['PCA','orthogonal','oblique']
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 def dn(x,v):
  s=(x.square().mean(-1)+eps).sqrt();return v/s[:,None]-x*(x*v).mean(-1)[:,None]/s[:,None]**3
 y0=native(p);st=p+y0;x=F.rms_norm(p,(1152,));lx=x@L.T;rx=x@R.T
 def tangent(v):return ((dn(p,v)@L.T)*rx+lx*(dn(p,v)@R.T))@D.T
 for head in cfg['heads']:
  B=torch.tensor(compiled[f'h{head}_basis'],device='cuda');cs=torch.tensor(compiled[f'h{head}_covariance'],device='cuda').sqrt();U=torch.tensor(compiled[f'h{head}_weighted_eigenvectors'][:,:4],device='cuda');h=torch.tensor(sources[f'h{head}_sources'],device='cuda');Q=torch.linalg.qr(B@(cs[:,None]*U)).Q;maps=[(h@B[:,:4])@B[:,:4].T,(h@Q)@Q.T,(((h@B/cs)@U)@U.T*cs)@B.T];hh=h[np.arange(512)%64].float();preds=[v[np.arange(512)%64].float() for v in maps];vs=[hh]+[hh-v for v in preds];dys=[tangent(v) for v in vs];local=[]
  # Independently audit the float32 MLP directional derivative against double algebra.
  pd=p[:1].double();vd=vs[0][:1].double();xd=pd/(pd.square().mean(-1)+eps).sqrt()[:,None];ld,rd,dd=L.double(),R.double(),D.double();dxd=dn(pd,vd);ref=((dxd@ld.T)*(xd@rd.T)+(xd@ld.T)*(dxd@rd.T))@dd.T;rel=float((ref-dys[0][:1].double()).norm()/ref.norm());assert rel<1e-4;audits[str(head)]={'MLP_tangent_float_relative':rel};del ld,rd,dd
  for start in range(0,512,16):
   sl=slice(start,start+16);base=lp(st[sl]);prob=base.exp();soft=torch.tanh(F.rms_norm(st[sl],(1152,))@W.T/30);cols=[]
   for dy in dys:
    J=((dn(st[sl],dy[sl])@W.T)*(1-soft.square())).double();J-=(prob*J).sum(-1)[:,None];cols.append(.5*(prob*J.square()).sum(-1))
   local.append(torch.stack(cols,-1).cpu().numpy())
  loc=np.concatenate(local);saved[f'h{head}_local']=loc;lv=loc.sum(0);metrics[str(head)]={'local_recovery':{n:float(1-lv[i+1]/lv[0]) for i,n in enumerate(labels)},'finite':{}}
  for scale in cfg['scales']:
   for sign,sgn in [('negative',-1),('positive',1)]:
    alpha=sgn*scale;yf=native(p-alpha*hh);ys=[native(p-alpha*v) for v in preds];rows=[]
    for start in range(0,512,16):
     sl=slice(start,start+16);tf=lp(p[sl]+yf[sl]);pf=tf.exp();cols=[(pf*(tf-lp(st[sl]))).sum(-1)];cols.extend((pf*(tf-lp(p[sl]+yy[sl]))).sum(-1) for yy in ys);rows.append(torch.stack(cols,-1).cpu().numpy())
    rr=np.concatenate(rows);name=f'h{head}_a{scale}_{sign}';saved[name]=rr;vv=rr.sum(0);metrics[str(head)]['finite'][f'{scale}_{sign}']={'recovery':{n:float(1-vv[i+1]/vv[0]) for i,n in enumerate(labels)},'KL_over_local_quadratic':{n:float(vv[i]/(scale**2*lv[i])) for i,n in enumerate(['effect']+labels)},'mean_effect_KL':float(vv[0]/512),'min_KL':float(rr.min())}
    if scale==1.0:
     prev=old[f'h{head}_{sign}'];po=oldortho[f'h{head}_{sign}'];target=np.stack([prev[:,0],prev[:,1],po[:,1],prev[:,2]],-1);drift=float(np.max(np.abs(rr.sum(0)-target.sum(0)))/vv[0]);assert drift<1e-5;audits[str(head)][sign+'_unit_replay_drift']=drift
  print(head,metrics[str(head)],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H502','config':cfg,'metrics':metrics,'audits':audits,'columns':['effect','PCA_error','orthogonal_error','oblique_error'],'local_units':'one-half probability-weighted centered-logit tangent squared norm','unique_documents':19456},indent=2)+'\n')
if __name__=='__main__':main()

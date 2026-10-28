import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_allheads_source_rank'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();_,dh,_=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),cfg);src=np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz');calids=src['calibration_ids'];donors=src['donors'];L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();p=torch.tensor(np.load(ART/'bilin18_l17_structured_metric_subspace_rows.npz')['pre'],device='cuda');previous=np.load(ART/'bilin18_l17_structured_allheads_rows.npz');saved={};metrics={};geometry={};audits={}
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p)
 for head in cfg['heads']:
  cal=dh[calids,head].double();mu=cal.mean(0);center=cal-mu;_,sv,Vh=torch.linalg.svd(center,full_matrices=False);h=dh[donors,head].double()-mu;hh=h[np.arange(512)%64];metrics[str(head)]={};geometry[str(head)]={}
  for k in cfg['source_ranks']:
   b=Vh[:k].T;hp=(hh@b)@b.T;geometry[str(head)][str(k)]={'calibration_energy':float(sv[:k].square().sum()/sv.square().sum()),'donor_energy':float(hp.square().sum()/hh.square().sum()),'source_relative_error':float((hh-hp).norm()/hh.norm())};metrics[str(head)][str(k)]={}
   if k==128:assert geometry[str(head)][str(k)]['source_relative_error']<1e-5
   for sign,alpha in [('negative',-1),('positive',1)]:
    yf=native(p-alpha*hh.float());yp=native(p-alpha*hp.float());uf=(yf-y0).double();up=(yp-y0).double();rows=[]
    for start in range(0,512,16):
     sl=slice(start,start+16);tf=lp(p[sl]+yf[sl]);tp=lp(p[sl]+yp[sl]);pb=lp(p[sl]+y0[sl]);pf=tf.exp();rows.append(torch.stack([(pf*(tf-pb)).sum(-1),(pf*(tf-tp)).sum(-1),uf[sl].square().sum(-1),(uf[sl]-up[sl]).square().sum(-1)],-1).cpu().numpy())
    rr=np.concatenate(rows);saved[f'h{head}_k{k}_{sign}']=rr;v=rr.sum(0);metrics[str(head)][str(k)][sign]={'mean_effect_KL':float(v[0]/512),'KL_recovery':float(1-v[1]/v[0]),'raw_recovery':float(1-v[3]/v[2])}
    if k==4:
     old=previous[f'h{head}_{sign}'];drift=float(abs(rr[:,1].sum()-old[:,2].sum())/v[0]);audits[f'h{head}_{sign}_rank4_recovery_drift']=drift;assert drift<1e-5
    if k==128:assert abs(1-metrics[str(head)][str(k)][sign]['KL_recovery'])<1e-4
   print('head',head,'rank',k,geometry[str(head)][str(k)],metrics[str(head)][str(k)],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H497','config':cfg,'metrics':metrics,'source_geometry':geometry,'audits':audits,'columns':['effect_KL','projection_error_KL','effect_raw_energy','projection_raw_error'],'unique_documents':18880},indent=2)+'\n')
if __name__=='__main__':main()

import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_fisher_orthogonal_control'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();old=np.load(ART/'bilin18_l17_fisher_source_donor_transfer_rows.npz');p=torch.tensor(old['pre'],device='cuda');sources=np.load(ART/'bilin18_l17_fisher_source_donor_transfer_sources.npz');compiled=np.load(ART/'bilin18_l17_fisher_source_rank_compiled.npz');L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();saved={};metrics={};geometry={};audits={}
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p)
 for head in cfg['heads']:
  B=torch.tensor(compiled[f'h{head}_basis'],device='cuda');cs=torch.tensor(compiled[f'h{head}_covariance'],device='cuda').sqrt();U=torch.tensor(compiled[f'h{head}_weighted_eigenvectors'][:,:4],device='cuda');h=torch.tensor(sources[f'h{head}_sources'],device='cuda');write=B@(cs[:,None]*U);orth=torch.linalg.qr(write).Q;ho=(h@orth)@orth.T;hf=(((h@B/cs)@U)@U.T*cs)@B.T;oe=(h-ho).square().sum();fe=(h-hf).square().sum();assert float(oe-fe)<1e-8*float(h.square().sum());geometry[str(head)]={'orthogonal_source_raw_recovery':float(1-oe/h.square().sum()),'oblique_source_raw_recovery':float(1-fe/h.square().sum())};audits[str(head)]={'orthogonality':float((orth.T@orth-torch.eye(4,device='cuda')).abs().max()),'write_span_relative_residual':float((write-orth@(orth.T@write)).norm()/write.norm())};assert max(audits[str(head)].values())<1e-10;hh=h[np.arange(512)%64];hp=ho[np.arange(512)%64];metrics[str(head)]={}
  for sign,alpha in [('negative',-1),('positive',1)]:
   yf=native(p-alpha*hh.float());yp=native(p-alpha*hp.float());uf=(yf-y0).double();up=(yp-y0).double();rows=[]
   for start in range(0,512,16):
    sl=slice(start,start+16);tf=lp(p[sl]+yf[sl]);tp=lp(p[sl]+yp[sl]);pb=lp(p[sl]+y0[sl]);pf=tf.exp();rows.append(torch.stack([(pf*(tf-pb)).sum(-1),(pf*(tf-tp)).sum(-1),uf[sl].square().sum(-1),(uf[sl]-up[sl]).square().sum(-1)],-1).cpu().numpy())
   rr=np.concatenate(rows);name=f'h{head}_{sign}';saved[name]=rr;v=rr.sum(0);prev=old[name];assert np.allclose(rr[:,0],prev[:,0],rtol=1e-5,atol=1e-9);metrics[str(head)][sign]={'orthogonal_KL_recovery':float(1-v[1]/v[0]),'PCA_KL_recovery':float(1-prev[:,1].sum()/v[0]),'oblique_KL_recovery':float(1-prev[:,2].sum()/v[0]),'orthogonal_minus_PCA':float((prev[:,1].sum()-v[1])/v[0]),'oblique_minus_orthogonal':float((v[1]-prev[:,2].sum())/v[0]),'orthogonal_raw_recovery':float(1-v[3]/v[2])}
  print(head,metrics[str(head)],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H501','config':cfg,'metrics':metrics,'geometry':geometry,'audits':audits,'columns':['effect_KL','orthogonal_error_KL','effect_raw','orthogonal_raw_error'],'unique_documents':19456},indent=2)+'\n')
if __name__=='__main__':main()

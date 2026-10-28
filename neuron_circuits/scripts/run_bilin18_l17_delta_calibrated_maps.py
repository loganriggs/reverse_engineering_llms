import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_delta_calibrated_maps'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 def T(x):return torch.tensor(x,device='cuda',dtype=torch.float64)
 source_names=cfg['calibration_documents'];caldocs=np.concatenate([np.load(ART/(n+'.npy')) for n in source_names]);assert len(caldocs)==3072
 trainhashes=set()
 for n in source_names:trainhashes|={x['document_token_sha256'] for x in json.loads((ART/(n+'.manifest.json')).read_text())['records']}
 testhashes={x['document_token_sha256'] for x in json.loads((ART/'bilin18_label_direction_confirmation_documents.manifest.json').read_text())['records']};assert len(trainhashes)==3072 and not trainhashes&testhashes
 sums=[]
 for length in [32,128]:
  pp,heads,err=collect_sources(model,np.ascontiguousarray(caldocs[:,128-length:128]),dict(cfg,rows=3072,positions=[length-1]));assert err<1e-5;sums.append(heads.double().sum(1));del heads;print('calibration collected',length,flush=True)
 train_delta=sums[1]-sums[0];z=np.load(ART/'bilin18_l17_source_sample_curve_models.npz');B=T(z['basis']);FM=T(z['F']);dc=train_delta@B;C=dc.T@dc/3072;ev,V=torch.linalg.eigh((C+C.T)/2);assert ev.min()>0;cs=(V*ev.sqrt())@V.T;ci=(V/ev.sqrt())@V.T;G=cs@FM@cs;_,U=torch.linalg.eigh((G+G.T)/2);U=U.flip(1);P=V.flip(1);newmaps={};compiled={'basis':B.cpu().numpy(),'C_delta':C.cpu().numpy(),'F':FM.cpu().numpy(),'training_delta':train_delta.cpu().numpy()};fit_audits={'calibration_documents':len(trainhashes),'evaluation_overlap':0,'covariance_condition':float(ev.max()/ev.min()),'root_inverse_error':float((cs@ci-torch.eye(1023,device='cuda')).abs().max())};assert fit_audits['root_inverse_error']<1e-7
 for k in cfg['ranks']:
  u=U[:,:k];v=P[:,:k]
  for method,A in [('PCA',v@v.T),('Fisher',cs@u@u.T@ci)]:
   name=method+str(k);assert float((A@A-A).norm()/A.norm())<1e-7;newmaps[name]=A;compiled[name]=A.cpu().numpy()
 np.savez_compressed(STEM.with_name(STEM.name+'_models.npz'),**compiled);print('UPDATE MAPS SAVED BEFORE EVALUATION',flush=True)
 src=np.load(ART/'bilin18_l17_history_transfer_sources.npz');p=torch.tensor(src['pre'][:,3],device='cuda');ss=T(src['sum_writes']);targets=torch.tensor(src['targets'][:,3],device='cuda');assert np.all(src['targets']==src['targets'][:,3,None])
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach()
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 y=native(p);names=['zero','identity']+[f'{method}{k}' for k in cfg['ranks'] for method in ['PCA','Fisher']];saved={};metrics={};audits=fit_audits
 for i,length in enumerate(cfg['short_lengths']):
  delta=ss[:,3]-ss[:,i];dc=delta@B;metrics[str(length)]={};saved[f'L{length}_delta_energy']=delta.square().sum(-1).cpu().numpy()
  for name in names:
   e=delta if name=='zero' else torch.zeros_like(delta) if name=='identity' else delta-dc@newmaps[name].T@B.T
   inp=p-e.float();yp=native(inp);rows=[]
   for start in range(0,2048,16):
    sl=slice(start,start+16);base=lp(p[sl]+y[sl]);q=lp(inp[sl]+yp[sl]);rows.append(torch.stack([(base.exp()*(base-q)).sum(-1),(base-q).gather(1,targets[sl,None]).squeeze(1)],-1).cpu().numpy())
   rr=np.concatenate(rows);assert np.isfinite(rr).all();saved[f'L{length}_{name}']=rr;metrics[str(length)][name]={'mean_KL':float(rr[:,0].mean()),'mean_CE_change':float(rr[:,1].mean()),'CE_change_SE':float(rr[:,1].std(ddof=1)/np.sqrt(2048)),'raw_delta_recovery':float(1-e.square().sum()/delta.square().sum())}
   if name=='identity':audits[f'L{length}_identity_max_absolute']=float(np.abs(rr).max());assert np.abs(rr).max()==0
  for name,m in metrics[str(length)].items():m['KL_recovery_vs_zero_delta']=1-m['mean_KL']/metrics[str(length)]['zero']['mean_KL']
  print(length,metrics[str(length)],flush=True)
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H532','config':cfg,'metrics':metrics,'audits':audits,'unique_documents':26112},indent=2)+'\n')
if __name__=='__main__':main()

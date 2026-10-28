import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_canonical_measure_factorial'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 pp=np.empty((512,1152),np.float32)
 for p in ['compact_operation_confirmation','unprojected_finite_confirmation','covariance_source_confirmation','covariance_fresh_confirmation']:
  p='bilin18_l17_'+p;ids=json.loads((ART/f'{p}_selection.json').read_text())['sites'];pp[ids]=np.load(ART/f'{p}_rows.npz')['pre']
 split=json.loads((ART/'bilin18_l17_data_canonical_pilot_selection.json').read_text());tr=split['train'];te=split['test'];p=torch.tensor(pp,device='cuda');h=torch.tensor(np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['sources'][0],device='cuda').float();trainh=h[np.arange(256)%32];testh=h[np.arange(128)%64];x0=F.rms_norm(p,(1152,));xa=torch.cat([x0[tr],F.rms_norm(p[tr]+trainh,(1152,)),F.rms_norm(p[tr]-trainh,(1152,))]);model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach()
 def native(x):return ((x@L.T)*(x@R.T))@D.T+bias
 ya=native(xa);scale=(ya[:256]-ya[:256].mean(0)).square().mean().sqrt();models={};save={};audit={};repeat_error=0.
 for measure,support in [('natural',xa[:256]),('augmented',xa)]:
  support=support.double();N=len(support);K=(support@support.T/1152).square();km=K.mean(0);grand=K.mean();kc=K-km[None]-km[:,None]+grand;ev,U=torch.linalg.eigh(kc);ev=ev.flip(0)[:64];U=U.flip(1)[:,:64];assert float(ev[-1])>1e-10;enc=U/ev[None]*N**.5
  if measure=='natural':
   repeated=kc.repeat(3,3);re=torch.linalg.eigvalsh(repeated).flip(0)[:64];repeat_error=float((re/768-ev/256).abs().max());assert repeat_error<1e-10
  def feat(xx):
   k=(xx.double()@support.T/1152).square();return (k-km[None]-k.mean(1,keepdim=True)+grand)@enc
  z=feat(xa);save[measure+'_support']=support.cpu().numpy();save[measure+'_kernel_mean']=km.cpu().numpy();save[measure+'_kernel_grand']=float(grand);save[measure+'_kernel_encoder']=enc.cpu().numpy()
  for fit,ids in [('natural',np.arange(256)),('augmented',np.arange(768))]:
   Z=z[ids];Y=ya[ids].double()/scale.double();zm=Z.mean(0);ym=Y.mean(0);zc=Z-zm;yc=Y-ym;G=zc.T@zc/len(ids);ge,gu=torch.linalg.eigh(G);keep=ge>ge.max()*cfg['gram_relative_cutoff'];invsqrt=(gu[:,keep]*ge[keep].rsqrt())@gu[:,keep].T;cross=zc.T@yc/len(ids);u,s,vh=torch.linalg.svd(invsqrt@cross,full_matrices=False);C=invsqrt@((u[:,:16]*s[:16])@vh[:16]);intercept=ym-zm@C;name=measure+'_'+fit;models[name]=(support,km,grand,enc,C.float(),intercept.float());save[name+'_map']=C.cpu().numpy();save[name+'_intercept']=intercept.cpu().numpy();audit[name]={'fit_gram_rank':int(keep.sum()),'fit_gram_min_eigenvalue':float(ge.min()),'fit_gram_max_eigenvalue':float(ge.max()),'train_fit_normalized_mse':float((Z@C+intercept-Y).square().mean())};print('fit',name,audit[name],flush=True)
 save['scale']=float(scale);np.savez_compressed(STEM.with_name(STEM.name+'_models.npz'),**save);print('all models frozen before test',flush=True)
 def predict(x,m):
  support,km,grand,enc,C,b=m;k=(x.double()@support.T/1152).square();z=((k-km[None]-k.mean(1,keepdim=True)+grand)@enc).float();return scale*(z@C+b)
 def lp(state):return (30*torch.tanh(F.rms_norm(state,(1152,))@W.T/30)).double().log_softmax(-1)
 labels=torch.tensor(np.load(ART/'bilin18_compact_operation_confirmation_documents.npy')[te,64].astype(np.int64),device='cuda');pt=p[te];y0=native(x0[te]);pred0={n:predict(x0[te],m) for n,m in models.items()};rows={};metrics={}
 for mode,alpha in [('intact',0),('negative',-1),('positive',1)]:
  xx=F.rms_norm(pt-alpha*testh,(1152,));yy=native(xx);pred={n:predict(xx,m) for n,m in models.items()};acc={n:[] for n in models};de={n:[] for n in models};effects=[]
  for start in range(0,128,16):
   sl=slice(start,start+16);truth=lp(pt[sl]+yy[sl]);prob=truth.exp();base=lp(pt[sl]+y0[sl]);labs=labels[sl];ar=torch.arange(len(labs),device='cuda');ce=-truth[ar,labs];effects.append(torch.stack([(prob*(truth-base)).sum(-1),(yy[sl]-y0[sl]).double().square().sum(-1)],-1).cpu().numpy())
   for n in models:
    q=lp(pt[sl]+pred[n][sl]);acc[n].append(torch.stack([((pred[n][sl]-yy[sl])/scale).square().mean(-1),(prob*(truth-q)).sum(-1),-q[ar,labs]-ce,(q.argmax(-1)==truth.argmax(-1)).float()],-1).cpu().numpy());delta=pred[n][sl]-pred0[n][sl];qd=lp(pt[sl]+y0[sl]+delta);de[n].append(torch.stack([(prob*(truth-qd)).sum(-1),(delta-(yy[sl]-y0[sl])).double().square().sum(-1)],-1).cpu().numpy())
  rows[mode+'_effects']=np.concatenate(effects)
  for n in models:
   rows[mode+'_'+n]=np.concatenate(acc[n]);rows[mode+'_'+n+'_delta']=np.concatenate(de[n]);metrics[mode+'_'+n]=dict(zip(['write_MSE','KL','CE_delta','top1_agreement'],rows[mode+'_'+n].mean(0).tolist()));print(mode,n,metrics[mode+'_'+n],flush=True)
 old=np.load(ART/'bilin18_l17_canonical_objective_transfer_rows.npz');replay=max(float(abs(rows[mode+'_natural_natural']-old[mode+'_write']).max()) for mode in ['intact','negative','positive']);assert replay<1e-4;assert all(np.isfinite(v).all() for v in rows.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);j={'experiment':'H469','config':cfg,'metrics':metrics,'fit_audits':audit,'repeat_spectrum_max_abs':repeat_error,'baseline_replay_max_abs':replay,'numerical_gates':{'repetition_invariance':repeat_error<1e-10,'baseline_replay':replay<1e-4},'unique_completed_documents':12992};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()

import json,hashlib,itertools,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_joint_pca_more_calibration'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();precal,heads,spliterror=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),dict(cfg,rows=1024));cal=heads.double().sum(1);z=np.load(ART/'bilin18_l17_joint_source_canonical_model.npz');mu=torch.tensor(z['mean'],device='cuda');old=np.load(ART/'bilin18_l17_joint_source_confirmation_rows.npz');p=torch.tensor(old['pre'],device='cuda');h=torch.tensor(old['head_writes'],device='cuda').double().sum(1)-mu;targets=torch.tensor(old['targets'],device='cuda');L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();names=['reference436','left512','right512','pooled1024'];bs={'reference436':torch.tensor(z['basis'][:,:36],device='cuda')};models={};geometry={};pairgeo={};metrics={};saved={}
 for name,xx in [('left512',cal[:512]),('right512',cal[512:]),('pooled1024',cal)]:
  mean=xx.mean(0);center=xx-mean;_,sv,vh=torch.linalg.svd(center,full_matrices=False);rank=int((sv>sv[0]*1e-10).sum());B=vh[:rank].T;bs[name]=B[:,:36];models[name+'_basis']=B.cpu().numpy();models[name+'_singular_values']=sv.cpu().numpy();models[name+'_covariance_mean']=mean.cpu().numpy();geometry[name]={'empirical_rank':rank,'heldout_outside_span_energy':float((h-(h@B)@B.T).square().sum()/h.square().sum()),'top36_calibration_energy':float(sv[:36].square().sum()/sv.square().sum())};print('fit',name,geometry[name],flush=True)
 pairs=list(itertools.combinations(names,2))
 for aa,bb in pairs:
  cos=torch.linalg.svdvals(bs[aa].T@bs[bb]).clamp(0,1);pairgeo[aa+'__'+bb]={'mean_cosine_squared':float(cos.square().mean()),'median_angle_degrees':float(torch.rad2deg(torch.acos(cos)).median()),'max_angle_degrees':float(torch.rad2deg(torch.acos(cos)).max())}
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p);inps={n:p-(h-(h@b)@b.T).float() for n,b in bs.items()};ys={n:native(inp) for n,inp in inps.items()};rows=[]
 for start in range(0,512,16):
  sl=slice(start,start+16);base=lp(p[sl]+y0[sl]);qs={n:lp(inps[n][sl]+ys[n][sl]) for n in names};cols=[]
  for n in names:cols.extend([(base.exp()*(base-qs[n])).sum(-1),(base-qs[n]).gather(1,targets[sl,None]).squeeze(1)])
  for aa,bb in pairs:cols.append(.5*((qs[aa].exp()*(qs[aa]-qs[bb])).sum(-1)+(qs[bb].exp()*(qs[bb]-qs[aa])).sum(-1)))
  rows.append(torch.stack(cols,-1).cpu().numpy())
 rr=np.concatenate(rows);assert np.allclose(rr[:,:2],old['joint_PCA'],rtol=1e-5,atol=1e-9);den=old['mean'][:,0].mean()
 for i,n in enumerate(names):metrics[n]={'mean_KL':float(rr[:,2*i].mean()),'recovery':float(1-rr[:,2*i].mean()/den),'mean_CE_change':float(rr[:,2*i+1].mean()),'CE_change_SE':float(rr[:,2*i+1].std(ddof=1)/np.sqrt(512))}
 for i,(aa,bb) in enumerate(pairs):pairgeo[aa+'__'+bb].update(mean_symmetric_KL=float(rr[:,8+i].mean()),pair_KL_over_mean_error=float(rr[:,8+i].mean()/((metrics[aa]['mean_KL']+metrics[bb]['mean_KL'])/2)))
 assert np.isfinite(rr).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),rows=rr);np.savez_compressed(STEM.with_name(STEM.name+'_models.npz'),**models,shared_reconstruction_mean=mu.cpu().numpy(),calibration_sum=cal.cpu().numpy(),calibration_pre=precal.cpu().numpy());STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H514','config':cfg,'metrics':metrics,'geometry':geometry,'pairs':pairgeo,'head_write_sum_relative':spliterror,'columns':[x for n in names for x in [n+'_KL',n+'_CE']]+[aa+'__'+bb+'_symmetric_KL' for aa,bb in pairs],'unique_documents':21504},indent=2)+'\n');print('metrics',metrics,flush=True);print('pair left/right',pairgeo['left512__right512'],flush=True)
if __name__=='__main__':main()

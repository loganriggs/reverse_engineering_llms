import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_source_split_stability'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 def T(x):return torch.tensor(x,device='cuda',dtype=torch.float64)
 z=np.load(ART/'bilin18_l17_metric_sample_curve_models.npz');B=T(z['basis']);mu=T(z['mean']);FM=T(z['F1024']);cal=np.concatenate([np.load(ART/'bilin18_l17_joint_pca_more_calibration_models.npz')['calibration_sum'],np.load(ART/'bilin18_l17_source_sample_curve_sources.npz')['new_sum']]);assert len(cal)==3072
 rng=np.random.default_rng(cfg['seed']);splits=np.stack([rng.permutation(3072).reshape(2,1536) for _ in range(4)]);assert all(np.array_equal(np.sort(s.ravel()),np.arange(3072)) for s in splits);np.save(STEM.with_name(STEM.name+'_splits.npy'),splits);cal=T(cal);maps={};writes={};compiled={'basis':B.cpu().numpy(),'mean':mu.cpu().numpy()};audits={};geometry={};eye=torch.eye(1023,device='cuda',dtype=torch.float64)
 for rep,split in enumerate(splits):
  for side,ids in enumerate(split):
   xx=cal[ids];xc=(xx-xx.mean(0))@B;C=xc.T@xc/1535;C=(C+C.T)/2;ev,V=torch.linalg.eigh(C);assert ev.min()>0;cs=(V*ev.sqrt())@V.T;ci=(V/ev.sqrt())@V.T;G=cs@FM@cs;fv,U=torch.linalg.eigh((G+G.T)/2);U=U.flip(1);P=V.flip(1);err=float((cs@ci-eye).abs().max());assert err<1e-7;audits[f'r{rep}s{side}']={'root_inverse_error':err,'condition':float(ev.max()/ev.min())}
   for k in cfg['ranks']:
    for method in ['PCA','Fisher']:
     w=P[:,:k] if method=='PCA' else cs@U[:,:k];A=w@w.T if method=='PCA' else w@U[:,:k].T@ci;name=f'r{rep}s{side}_{method}{k}';maps[name]=A;writes[name]=torch.linalg.qr(w,mode='reduced')[0];compiled[name]=A.cpu().numpy();assert float((A@A-A).norm()/A.norm())<1e-7
  for k in cfg['ranks']:
   for method in ['PCA','Fisher']:
    aa=f'r{rep}s0_{method}{k}';bb=f'r{rep}s1_{method}{k}';cos=torch.linalg.svdvals(writes[aa].T@writes[bb]).clamp(0,1);A,D=maps[aa],maps[bb];geometry[f'r{rep}_{method}{k}']={'write_mean_cosine_squared':float(cos.square().mean()),'write_median_angle_degrees':float(torch.rad2deg(torch.acos(cos)).median()),'write_max_angle_degrees':float(torch.rad2deg(torch.acos(cos)).max()),'relative_operator_difference':float((A-D).norm()/torch.sqrt((A.square().sum()+D.square().sum())/2))}
  print('fit split',rep,flush=True)
 np.savez_compressed(STEM.with_name(STEM.name+'_models.npz'),**compiled);print('ALL MAPS SAVED BEFORE EVALUATION',flush=True)
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach()
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 saved={};metrics={}
 for panel,file in cfg['evaluation_panels'].items():
  data=np.load(ART/file);p=torch.tensor(data['pre'],device='cuda');h=T(data['head_writes']).sum(1)-mu;hc=h@B;y=native(p);den=float(data['mean'][:,0].mean());metrics[panel]={}
  for rep in range(4):
   for k in cfg['ranks']:
    for method in ['PCA','Fisher']:
     name=f'r{rep}_{method}{k}';aa=maps[f'r{rep}s0_{method}{k}'];bb=maps[f'r{rep}s1_{method}{k}'];ia=p-(h-hc@aa.T@B.T).float();ib=p-(h-hc@bb.T@B.T).float();sa=ia+native(ia);sb=ib+native(ib);rows=[]
     for start in range(0,512,16):
      sl=slice(start,start+16);base=lp(p[sl]+y[sl]);qa=lp(sa[sl]);qb=lp(sb[sl]);rows.append(torch.stack([(base.exp()*(base-qa)).sum(-1),(base.exp()*(base-qb)).sum(-1),.5*((qa.exp()*(qa-qb)).sum(-1)+(qb.exp()*(qb-qa)).sum(-1))],-1).cpu().numpy())
     rr=np.concatenate(rows);assert np.isfinite(rr).all();saved[panel+'_'+name]=rr;m=rr.mean(0);metrics[panel][name]={'left_KL':float(m[0]),'right_KL':float(m[1]),'left_recovery':float(1-m[0]/den),'right_recovery':float(1-m[1]/den),'symmetric_KL':float(m[2]),'disagreement_over_average_error':float(m[2]/m[:2].mean()),'disagreement_over_mean_removal':float(m[2]/den)}
   print('evaluated',panel,'split',rep,flush=True)
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H523','config':cfg,'geometry':geometry,'metrics':metrics,'audits':audits,'unique_documents':24064},indent=2)+'\n')
if __name__=='__main__':main()

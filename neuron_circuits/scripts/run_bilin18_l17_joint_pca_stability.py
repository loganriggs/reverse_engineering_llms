import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_joint_pca_stability'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();_,heads,spliterror=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),cfg);ids=np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['calibration_ids'];cal=heads[ids].double().sum(1);assert len(cal)==436;z=np.load(ART/'bilin18_l17_joint_source_canonical_model.npz');mu=torch.tensor(z['mean'],device='cuda');old=np.load(ART/'bilin18_l17_joint_source_confirmation_rows.npz');p=torch.tensor(old['pre'],device='cuda');h=torch.tensor(old['head_writes'],device='cuda').double().sum(1)-mu;L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();B0=torch.tensor(z['basis'][:,:36],device='cuda');saved={};bases={};metrics={};geometry={};splitids={};rng=np.random.default_rng(cfg['split_seed'])
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p);reference_inp=p-(h-(h@B0)@B0.T).float();yr=native(reference_inp)
 for rep in range(4):
  perm=rng.permutation(436);parts=[perm[:218],perm[218:]];assert not set(parts[0])&set(parts[1]);splitids[str(rep)]=[ids[v].tolist() for v in parts];bs=[];inps=[];ys=[]
  for side,ii in enumerate(parts):
   half=cal[ii];_,sv,vh=torch.linalg.svd(half-half.mean(0),full_matrices=False);b=vh[:36].T;bs.append(b);bases[f'r{rep}_s{side}_basis']=b.cpu().numpy();bases[f'r{rep}_s{side}_sv']=sv.cpu().numpy();inp=p-(h-(h@b)@b.T).float();inps.append(inp);ys.append(native(inp))
  cos=torch.linalg.svdvals(bs[0].T@bs[1]).clamp(0,1);geometry[str(rep)]={'mean_principal_cosine_squared':float(cos.square().mean()),'median_angle_degrees':float(torch.rad2deg(torch.acos(cos)).median()),'max_angle_degrees':float(torch.rad2deg(torch.acos(cos)).max()),'principal_cosines':cos.cpu().tolist()};rows=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);base=lp(p[sl]+y0[sl]);ref=lp(reference_inp[sl]+yr[sl]);qa=lp(inps[0][sl]+ys[0][sl]);qb=lp(inps[1][sl]+ys[1][sl]);prob=base.exp();cols=[(prob*(base-ref)).sum(-1),(prob*(base-qa)).sum(-1),(prob*(base-qb)).sum(-1),.5*((qa.exp()*(qa-qb)).sum(-1)+(qb.exp()*(qb-qa)).sum(-1))];rows.append(torch.stack(cols,-1).cpu().numpy())
  rr=np.concatenate(rows);saved[f'r{rep}']=rr;assert np.allclose(rr[:,0],old['joint_PCA'][:,0],rtol=1e-5,atol=1e-9);v=rr.mean(0);den=old['mean'][:,0].mean();metrics[str(rep)]={'mean_KL_reference':float(v[0]),'mean_KL_left':float(v[1]),'mean_KL_right':float(v[2]),'recovery_left':float(1-v[1]/den),'recovery_right':float(1-v[2]/den),'mean_pair_symmetric_KL':float(v[3]),'pair_KL_over_mean_replacement_effect':float(v[3]/den),'pair_KL_over_mean_half_error':float(v[3]/((v[1]+v[2])/2))};print(rep,geometry[str(rep)],metrics[str(rep)],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);np.savez_compressed(STEM.with_name(STEM.name+'_bases.npz'),**bases);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H513','config':cfg,'metrics':metrics,'geometry':geometry,'split_document_ids':splitids,'head_write_sum_relative':spliterror,'columns':['reference_error_KL','left_error_KL','right_error_KL','pair_symmetric_KL'],'unique_documents':20480},indent=2)+'\n')
if __name__=='__main__':main()

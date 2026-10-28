import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_source_context_pairing'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();data=np.load(ART/'bilin18_l17_joint_source_confirmation_rows.npz');p=torch.tensor(data['pre'],device='cuda');z=np.load(ART/'bilin18_l17_joint_fisher_rank_curve_model.npz');B=torch.tensor(z['basis'],device='cuda');mu=torch.tensor(z['mean'],device='cuda');cs=torch.tensor(z['covariance'],device='cuda').sqrt();U=torch.tensor(z['U'],device='cuda');h=torch.tensor(data['head_writes'],device='cuda').double().sum(1)-mu;hc=h@B;errors={'mean':h.float()}
 for k in cfg['ranks']:
  errors[f'PCA{k}']=(h-hc[:,:k]@B[:,:k].T).float();u=U[:,:k];errors[f'Fisher{k}']=(h-(((hc/cs)@u)@u.T*cs)@B.T).float()
 rng=np.random.default_rng(cfg['permutation_seed']);perms=np.stack([np.arange(512)]+[rng.permutation(512) for _ in range(cfg['permutations'])]);assert all(np.array_equal(np.sort(q),np.arange(512)) for q in perms);np.savez_compressed(STEM.with_name(STEM.name+'_permutations.npz'),permutations=perms);L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();eps=torch.finfo(torch.float32).eps;names=list(errors);local=np.zeros((9,512,len(names)));error_energy=np.stack([errors[n].double().square().sum(-1).cpu().numpy() for n in names],-1)
 def dn(x,v):
  ss=(x.square().mean(-1)+eps).sqrt();return v/ss[:,None]-x*(x*v).mean(-1)[:,None]/ss[:,None]**3
 for pi,perm in enumerate(perms):
  for start in range(0,512,16):
   sl=slice(start,start+16);pp=p[sl];x=F.rms_norm(pp,(1152,));lx=x@L.T;rx=x@R.T;state=pp+(lx*rx)@D.T+bias;soft=torch.tanh(F.rms_norm(state,(1152,))@W.T/30);prob=(30*soft).double().softmax(-1)
   for ni,n in enumerate(names):
    e=errors[n][perm[sl]];dx=dn(pp,e);ds=e+((dx@L.T)*rx+lx*(dx@R.T))@D.T;J=((dn(state,ds)@W.T)*(1-soft.square())).double();J-=(prob*J).sum(-1)[:,None];local[pi,sl,ni]=(.5*(prob*J.square()).sum(-1)).cpu().numpy()
  print('pairing',pi,'local mean energies',local[pi].mean(0).tolist(),flush=True)
 agg=local.mean(1);metrics={}
 for pi in range(9):metrics[str(pi)]={'mean_local_KL':dict(zip(names,agg[pi].tolist())),'local_recovery':{n:float(1-agg[pi,i]/agg[pi,0]) for i,n in enumerate(names)},'Fisher_minus_PCA_recovery':{str(k):float((agg[pi,names.index(f'PCA{k}')]-agg[pi,names.index(f'Fisher{k}')])/agg[pi,0]) for k in cfg['ranks']}}
 penergy=p.double().square().sum(-1).cpu().numpy();corr={n:float(np.corrcoef(np.log(error_energy[:,i]),np.log(penergy))[0,1]) for i,n in enumerate(names)};assert np.isfinite(local).all() and local.min()>=0;np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),local_KL=local,source_error_energy=error_energy,receiver_energy=penergy,permutations=perms);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H517','config':cfg,'names':names,'metrics':metrics,'source_error_receiver_logenergy_correlation':corr,'unique_documents':21504},indent=2)+'\n')
if __name__=='__main__':main()

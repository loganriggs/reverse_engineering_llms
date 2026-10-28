"""H358 exact log-pairing split into actual and control metric placement."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM='bilin18_l17_head_context_selectivity'
def main():
 cfg=json.loads((ROOT/'configs'/f'{STEM}.json').read_text());assert hashlib.sha256((ART/cfg['source']).read_bytes()).hexdigest()==cfg['sha256'];ray=np.load(ART/cfg['source'])['rayleigh'];old=json.loads((ART/'bilin18_l17_allhead_native_pairing.json').read_text());rng=np.random.default_rng(cfg['bootstrap_seed']);docs=rng.integers(0,128,(4000,128));ranks=rng.integers(0,6,(4000,6));draws={n:rng.integers(0,6,(4000,6)) for n in ['moment','donor']}
 def placement(v):return np.log(v[:,0]).mean((1,2))-np.log(v[:,1:]).mean((1,2,3))
 A=ray[:,:,0];C={n:ray[:,:,start:start+6] for n,start in [('moment',1),('donor',7)]};est=np.stack([placement(A)]+[placement(C[n].mean(2)) for n in C]);bs=[]
 for k,ix in enumerate(docs):
  mr=np.r_[0,1+ranks[k]];aa=A[:,mr][:,:,ix];vals=[placement(aa)]
  for n in C:cc=C[n][:,:,draws[n][k]].mean(2);vals.append(placement(cc[:,mr][:,:,ix]))
  bs.append(np.stack(vals))
 bs=np.stack(bs);radius=float(np.quantile(np.abs(bs-est).max((1,2)),.95));summary={n:{'placement_gain':np.exp(est[i]).tolist(),'joint27_ci95':np.stack([np.exp(est[i]-radius),np.exp(est[i]+radius)],-1).tolist()} for i,n in enumerate(['actual','moment','donor'])}
 replay=max(np.max(np.abs(np.exp(est[i+1]-est[0])-np.array(old['summary'][n]['own_over_matched']))) for i,n in enumerate(C));assert replay<1e-10
 delta=est[:,6]-est[:,0];bd=bs[:,:,6]-bs[:,:,0];rad=float(np.quantile(np.abs(bd-delta).max(1),.95));contrast={n:{'head6_over_head0':float(np.exp(delta[i])),'joint3_ci95':np.exp([delta[i]-rad,delta[i]+rad]).tolist()} for i,n in enumerate(['actual','moment','donor'])};gates={'actual6_lower':contrast['actual']['joint3_ci95'][1]<.8,'controls_equivalent':all(contrast[n]['joint3_ci95'][0]>=.8 and contrast[n]['joint3_ci95'][1]<=1.25 for n in C)}
 np.savez_compressed(ART/f'{STEM}_bootstrap.npz',placement_log=bs);(ART/f'{STEM}.json').write_text(json.dumps({'config':cfg,'summary':summary,'contrast':contrast,'gates':gates,'pairing_identity_error':float(replay),'joint27_radius':radius,'joint3_radius':rad},indent=2)+'\n');print(summary,contrast,gates)
if __name__=='__main__':main()

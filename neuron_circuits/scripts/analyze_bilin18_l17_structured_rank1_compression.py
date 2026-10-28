from pathlib import Path
import json,numpy as np
A=Path(__file__).resolve().parents[1]/'artifacts/bilin18_tensor_decomposition';S=A/'bilin18_l17_structured_rank1_compression';j=json.loads(S.with_suffix('.json').read_text());z=np.load(S.with_name(S.name+'_rows.npz'));rng=np.random.default_rng(j['config']['bootstrap_seed']);ix=(rng.integers(0,8,(4000,64,8))*64+np.arange(64)[None,:,None]).reshape(4000,512);est=[];draw=[];names=[];threshold=[]
for sign in ['negative','positive']:
 e=z[sign+'_effects'][:,0];v=z[sign+'_structured'][:,0];ab=z[sign+'_omit_norm_correction'][:,0]
 names.extend(['absolute_'+sign,'norm_gain_'+sign]);threshold.extend([.95,0]);est.extend([1-v.sum()/e.sum(),(ab-v).sum()/e.sum()]);draw.extend([1-v[ix].sum(1)/e[ix].sum(1),(ab-v)[ix].sum(1)/e[ix].sum(1)])
est=np.array(est);draw=np.stack(draw,-1);sd=draw.std(0,ddof=1);q=np.quantile(np.max(abs((draw-est)/sd),axis=1),.95);j['primary_statistics']={n:{'estimate':float(v),'joint95_interval':[float(v-q*s),float(v+q*s)],'threshold':t,'pass':bool(v-q*s>t)} for n,v,s,t in zip(names,est,sd,threshold)};j['bootstrap_scope']='4000 paired within64fixed donor strata, eight receivers each, joint4 standardized bands; conditional on fitting/calibration/source pool';S.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print(j['primary_statistics'])

"""Post-result algebraic lower bound on contribution energy at the fitted function."""
import json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation';r=json.loads((P/'capacity_toys.json').read_text());saved=torch.load('/tmp/emlp_t2_models.pt',weights_only=True);rows=[]
for s in saved:
 st={k:v.double().numpy() for k,v in s['state'].items()};x=s['clean_inputs'].double().numpy();h=(x@st['L'].T)*(x@st['R'].T);net=h@st['D'].T;sv=np.linalg.svd(net,compute_uv=False);bound=float(sv.sum()**2/(8*np.square(sv).sum()));observed=r['rows'][s['row']]['geometry']['channel_net_energy_ratio'];assert observed+1e-6>=bound
 rows.append(dict(row=s['row'],observed_ratio=observed,nuclear_norm_bound=bound,gap=observed-bound))
(P/'capacity_geometry_bound.json').write_text(json.dumps(dict(protocol='Post-result inequality, not fitted. Each channel contribution matrix has rank1. Nuclear(Y)<=sum_i Frobenius(C_i)<=sqrt(width*sum_i Frobenius(C_i)^2). Therefore channel/net energy ratio >= nuclear(Y)^2/(width*Frobenius(Y)^2). Bias excluded.',rows=rows),indent=2)+'\n')
for c in (4,16):
 ids=[i for i,x in enumerate(r['rows']) if x['classes']==c and x['train_noise']==0 and x['penalty']==.01]
 print(c,'observed',np.mean([rows[i]['observed_ratio'] for i in ids]),'bound',np.mean([rows[i]['nuclear_norm_bound'] for i in ids]))

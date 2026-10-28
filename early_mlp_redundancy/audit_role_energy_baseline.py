"""M14 independent least-squares fit and prediction replay."""
import json,hashlib
from pathlib import Path
import torch
P=Path(__file__).resolve().parent/'validation';torch.set_num_threads(4)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ip=P/'role_energy_baseline.json';r=json.loads(ip.read_text());pp=P/'role_energy_protocol.json';protocol=json.loads(pp.read_text());assert sha(pp)==r['protocol_sha256'];mp=P/'group_role_matching.json';m=json.loads(mp.read_text());assert sha(mp)==protocol['matching_sha256'];errors=[]
for row in r['results']:
 prefix=f"{row['kind']}:{row['grouping']}:{row['seed']}:";tr=m['signatures'][prefix+'train'];te=m['signatures'][prefix+'evaluation'];E=torch.tensor(tr['energy'],dtype=torch.float64);Y=torch.tensor(tr['margin'],dtype=torch.float64);Et=torch.tensor(te['energy'],dtype=torch.float64);Yt=torch.tensor(te['margin'],dtype=torch.float64)
 if row['method']=='shared_energy':coef=torch.linalg.lstsq(E.reshape(-1,1),Y.reshape(-1,1)).solution.flatten();pred=coef[0]*Et
 elif row['method']=='group_energy':coef=torch.stack([torch.linalg.lstsq(e[:,None],y[:,None]).solution[0,0] for e,y in zip(E,Y)]);pred=coef[:,None]*Et
 else:coef=torch.linalg.lstsq(torch.ones(10,1,dtype=torch.float64),Y.T).solution.flatten();pred=coef[:,None].expand_as(Yt)
 saved=torch.tensor(row['coefficients'],dtype=torch.float64);err=float((coef-saved).abs().max()/saved.abs().max().clamp_min(1e-20));errors.append(err);assert err<1e-10;assert torch.allclose(pred,torch.tensor(row['prediction'],dtype=torch.float64),rtol=1e-10,atol=1e-12);target=(Yt*Yt).sum(0);sq=((pred-Yt)**2).sum(0);relative=float(sq.sum()/target.sum());assert abs(relative-row['relative_squared_error'])<1e-10;assert abs(float((sq/target).mean())-row['equal_family_mean_relative_error'])<1e-10;keep=[i for i,f in enumerate(m['families']['evaluation']) if f!='antonym'];assert abs(float(sq[keep].sum()/target[keep].sum())-row['relative_error_without_antonym'])<1e-10
for ref in r['M13_reference']:
 x=next(x for x in m['matches'] if all(x[k]==ref[k] for k in ['kind','grouping','method']));assert ref['relative_squared_error']==x['evaluation_relative_squared_error']
out=dict(status='passed',least_squares_fits=18,coefficients_and_predictions_checked=True,M13_references_unchanged=True,max_relative_coefficient_error=max(errors),result_sha256=sha(ip),model_forwards=0,scope='Torch64 least-squares versus NumPy closed-form coefficients; evaluation and concentration scores independently replayed. No newdata confirmation.')
(P/'role_energy_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

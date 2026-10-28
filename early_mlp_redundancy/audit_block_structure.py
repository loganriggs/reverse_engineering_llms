"""M12 independent direct-factor SVD and geometry/support checks."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
P=Path(__file__).resolve().parent/'validation';torch.set_num_threads(4)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rp=P/'block_structure.json';r=json.loads(rp.read_text());ip=Path(r['inputs_path']);assert sha(ip)==r['inputs_sha256'];assert sha(P/'block_structure_protocol.json')==r['protocol_sha256'];states=torch.load(ip,weights_only=True);states={s['key']:s for s in states};rankerr=[];spectralerr=[];rotationerr=[]
for row in r['rows']:
 s=states[row['key']];atoms=s['groups'][row['grouping']][row['group']];ix=slice(0,4096) if row['split']=='train' else slice(4096,8192);code=s['code'][ix][:,atoms];active=(s['support'][ix][:,atoms]>0).any(-1);a=code[active];assert len(a)==row['active_rows'];a=a-a.mean(0);D=s['D'][:,atoms];_,sing,vh=torch.linalg.svd(a,full_matrices=False);factor=sing[:,None]*vh@D.T;vals=torch.linalg.svdvals(factor).square()/len(a);record=torch.tensor(row['output_eigenvalues'][::-1],dtype=torch.float64);err=float((vals-record).abs().max()/vals.max().clamp_min(1e-20));spectralerr.append(err);assert err<1e-9;rank=float(vals.sum().square()/vals.square().sum());rankerr.append(abs(rank-row['conditional_participation_rank']));assert rankerr[-1]<1e-9;assert int(torch.searchsorted(vals.cumsum(0)/vals.sum(),.9))+1==row['conditional_r90']
 if row['group']==0 and row['grouping']=='contiguous' and row['split']=='evaluation':
  g=torch.Generator().manual_seed(12013);rotation=torch.linalg.qr(torch.randn(4,4,generator=g,dtype=torch.float64)).Q;original=a@D.T;rotated=(a@rotation)@(D@rotation).T;rotationerr.append(float((rotated-original).norm()/original.norm()));assert rotationerr[-1]<1e-12
rng=np.random.default_rng(12012);perm=rng.permutation(1152);sign=rng.choice([-1.,1.],size=1152);overlaperr=[]
for row in r['stability']:
 a=states[f"{row['kind']}:71:{row['arm']}"];b=states[f"{row['kind']}:72:{row['arm']}"];grouping='contiguous' if row['grouping']=='ambient_null' else row['grouping'];Db=b['D'].numpy();Db=Db[perm]*sign[:,None] if row['grouping']=='ambient_null' else Db;ia=a['groups'][grouping].numpy();ib=b['groups'][grouping].numpy();qa=np.stack([np.linalg.svd(a['D'].numpy()[:,ii],full_matrices=False)[0] for ii in ia]);qb=np.stack([np.linalg.svd(Db[:,ii],full_matrices=False)[0] for ii in ib]);overlap=np.square(np.einsum('gdi,hdj->ghij',qa,qb)).sum((2,3))/4;err=float(np.max(np.abs(overlap-np.array(row['overlap_matrix']))));overlaperr.append(err);assert err<1e-10;left,right=linear_sum_assignment(-overlap);assert sorted(row['assignment'])==list(range(32));assert abs(overlap[left,right].mean()-row['mean_matched_overlap'])<1e-10
 for split in ['train','evaluation']:
  ix=slice(0,4096) if split=='train' else slice(4096,8192);sa=np.stack([np.any(a['support'].numpy()[ix][:,ii]>0,axis=1) for ii in ia],1);sb=np.stack([np.any(b['support'].numpy()[ix][:,ii]>0,axis=1) for ii in ib],1);assignment=np.array(row['assignment']);inter=np.logical_and(sa,sb[:,assignment]).sum(0);union=np.logical_or(sa,sb[:,assignment]).sum(0);jac=np.divide(inter,union,out=np.zeros(32),where=union>0);assert np.max(np.abs(jac-np.array(row['supports'][split]['per_group_jaccard'])))<1e-12
for summary in r['summaries']:
 rows=[x for x in r['rows'] if all(x[k]==summary[k] for k in ['kind','seed','arm','grouping','split'])];assert len(rows)==32
 for key,field,func in [('mean_occupancy','occupancy',np.mean),('median_participation_rank','conditional_participation_rank',np.median),('median_r90','conditional_r90',np.median),('mean_conditional_all4_fraction','conditional_all4_fraction',np.mean)]:assert abs(float(func([x[field] for x in rows]))-summary[key])<1e-12
out=dict(status='passed',group_covariance_spectra=len(r['rows']),stability_matrices=len(r['stability']),rotation_checks=len(rotationerr),max_spectral_relative_error=max(spectralerr),max_rank_absolute_error=max(rankerr),max_overlap_absolute_error=max(overlaperr),max_rotation_relative_error=max(rotationerr),result_sha256=sha(rp),scope='Direct-factor Torch64SVD vsQRcovariance, SVD bases vsQRprincipaloverlap, support matching andsummary aggregation. No nativeforward/causal-role test.')
(P/'block_structure_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

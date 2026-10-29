"""M66 independent NumPy channel energies, expansions and subset formulas."""
from pathlib import Path
import hashlib,json,itertools,torch,numpy as np
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def en(x):return float(np.square(x).sum())
def main():
 torch.set_num_threads(1);rp=V/'native_change_coordination.json';r=json.loads(rp.read_text());assert r['status']=='completed' and r['source_sha256']==sha(P/'probe_native_change_coordination.py');wp=Path(r['weights_path']);mp=Path(r['masks_path']);assert sha(wp)==r['weights_sha256'] and sha(mp)==r['masks_sha256'];weights=torch.load(wp,weights_only=True);masks=torch.load(mp,weights_only=True).numpy();N=masks.shape[1];assert np.array_equal(masks.sum(1),np.full(32,N//2));W={l:{k:v.double().numpy() for k,v in ws.items()} for l,ws in weights.items()};n=0;maxrel=0.;mcmaxrel=0.;syn=[]
 # Exhaust all fixed-size subsets in geometrically distinct known cases.
 rng=np.random.default_rng(6602)
 for name,v in [('random',rng.normal(size=(6,4))),('duplicate',np.tile([[1.,2.,0.,0.]],(6,1))),('cancel',np.array([[1.,0.,0.,0.],[-1.,0.,0.,0.]]*3)),('orthogonal',np.eye(6))]:
  nchan=len(v);S=en(v.sum(0));C=en(v)
  for k in [1,3,5]:
   observed=[];double=[]
   for inds in itertools.combinations(range(nchan),k):
    kept=v[list(inds)].sum(0);observed.append(en(kept-v.sum(0)));double.append(en(2*kept-v.sum(0)))
   p=k/nchan;prediction=(1-p)**2*S+p*(1-p)*(nchan*C-S)/(nchan-1);assert np.isclose(np.mean(observed),prediction,rtol=1e-12,atol=1e-12)
   if k*2==nchan:assert np.isclose(np.mean(double),(nchan*C-S)/(nchan-1),rtol=1e-12,atol=1e-12)
   syn.append(dict(case=name,k=k,prediction=prediction,enumerated=float(np.mean(observed))))
 for file in r['files']:
  bp=Path(file['path']);assert sha(bp)==file['sha256'];b=torch.load(bp,weights_only=True)
  for spec in b['conditions']:
   row=next(x for x in r['rows'] if all(x[k]==spec[k] for k in ['pair_id','split','endpoint','layer','region']));ep=spec['endpoint'];l=spec['layer'];pos=spec['recipient_positions'];dp=spec['donor_positions'];cr=b['captures'][ep][l];cd=b['captures'][1-ep][l];w=W[l];D=w['Down'];zr=cr['normalized_input'][:,pos].double().numpy();zd=cd['normalized_input'][:,dp].double().numpy();c=cd['products'][:,dp].double().numpy()-cr['products'][:,pos].double().numpy();native=cd['write'][:,dp].double().numpy()-cr['write'][:,pos].double().numpy();delta=c@D.T;S=en(delta);ce=(c*c).sum((0,1))*(D*D).sum(0);C=float(ce.sum());dz=zd-zr;lr=zr@w['Left'].T;rr=zr@w['Right'].T;dl=dz@w['Left'].T;dr=dz@w['Right'].T;lin=(lr*dr+dl*rr)@D.T;quad=(dl*dr)@D.T;dlin=((zd@w['Left'].T)*dr+dl*(zd@w['Right'].T))@D.T;ne=en(native)
   vals=dict(S=S,C=C,cross_sum=S-C,channel_participation=C*C/en(ce) if C else None,C_over_S=C/S if S else None,native_delta_energy=ne,sum_relative_squared_error=en(delta-native)/ne if ne else None,expansion_relative_squared_error=en(lin+quad-delta)/S if S else None,donor_expansion_relative_squared_error=en(dlin-quad-delta)/S if S else None,linear_energy=en(lin),quadratic_energy=en(quad),linear_only_error=en(lin-delta),donor_linear_energy=en(dlin),donor_linear_only_error=en(dlin-delta),predicted_raw_half_error=.25*S+.25*(N*C-S)/(N-1),predicted_doubled_half_error=(N*C-S)/(N-1))
   for k,v in vals.items():
    if v is None:assert row[k] is None;continue
    maxrel=max(maxrel,abs(row[k]-v)/max(1.,abs(v)));assert np.isclose(row[k],v,rtol=2e-8,atol=1e-7),(k,row[k],v)
   kept=(c[None]*masks[:,None,None,:])@D.T;raw=np.square(kept-delta).sum((1,2,3));dbl=np.square(2*kept-delta).sum((1,2,3))
   for k,values in [('mc_raw',raw),('mc_doubled',dbl)]:
    actual=spec[k+'_errors'].numpy();mcmaxrel=max(mcmaxrel,float(np.max(np.abs(actual-values)/np.maximum(1.,np.abs(values)))));assert np.allclose(actual,values,rtol=2e-8,atol=1e-6);assert np.isclose(row[k+'_mean'],values.mean(),rtol=2e-8,atol=1e-6);assert np.isclose(row[k+'_se'],values.std(ddof=1)/32**.5,rtol=2e-8,atol=1e-6)
   n+=1
  if (r['files'].index(file)+1)%10==0:print('AUDIT_PAIRS',r['files'].index(file)+1,flush=True)
 assert n==1506 and sum(x['status']=='ineligible' for x in r['rows'])==6
 op=V/'native_change_coordination_audit.json';assert not op.exists();op.write_text(json.dumps(dict(status='passed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),conditions=n,ineligible_conditions=6,synthetic_checks=syn,max_scaled_metric_difference=maxrel,max_scaled_mc_difference=mcmaxrel,scope='Independent NumPy metrics/finite expansions and every32-mask error vector; exact small-system subset enumeration. Numerical disagreement with native float32 forward remains in producer rows. Monte Carlo is finite sampling, not exact expectation or behavioral evidence.'),indent=2)+'\n');print('PASS',n,maxrel,mcmaxrel,flush=True)
if __name__=='__main__':main()

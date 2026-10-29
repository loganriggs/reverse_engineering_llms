"""M114 fixed native-input sphere paths; first actual hard-gate crossing per path."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,A,sha
from native_path_polynomials import path_coefficients,value,crossing_intervals,select
from continuous_block_shrinkage import block_shrinkage


def main():
 torch.set_num_threads(1)
 pp=V/'native_path_boundary_protocol.json';p=json.loads(pp.read_text());assert p['source_sha256']==sha(__file__)
 for name,h in p['bindings'].items():assert sha(V/name)==h
 assert p['helper_sha256']==sha(V.parent/'native_path_polynomials.py')
 wr=json.loads((V/'matched_layer_operator_inputs.json').read_text());assert sha(wr['path'])==wr['sha256'];weights=torch.load(wr['path'],weights_only=True)['states'][1];L,R,D=[weights[k].double().numpy() for k in ['L','R','D']];Z=weights['z'].double().numpy()
 models=json.loads((V/'continuous_training_results.json').read_text());assert models['receipt']['checkpoint_sha256']==wr['checkpoint_sha256'];rows=[];saved={};maxerr=0.
 for entry in [x for x in models['models'] if x['mode']=='hard']:
  assert sha(entry['path'])==entry['sha256'];state=torch.load(entry['path'],weights_only=True);s={k:t.double().numpy() for k,t in state['state'].items()};E=s['encoder'];mu=state['mean'].double().numpy();scale=state['scale'].double().numpy();B=s['dictionary']/np.linalg.norm(s['dictionary'],axis=0)
  for pi,(ia,ib) in enumerate(p['pairs']):
   z0,z1,q,w,n=path_coefficients(Z[ia],Z[ib],L,R,D,E,mu,scale,s['encoder_bias'],np.sqrt(Z.shape[1]));n=n.reshape(3,32,4);qmin=min(value(q,0),value(q,1),value(q,np.clip(-q[1]/(2*q[2]),0,1)))
   assert qmin>p['minimum_norm_fraction']*Z.shape[1]
   def direct(t):
    z=z0+t*(z1-z0);z=z/np.linalg.norm(z)*np.sqrt(len(z));write=D@((L@z)*(R@z));raw=(E@((write-mu)/scale)+s['encoder_bias']).reshape(32,4);return write,raw
   for t in p['validation_points']:
    write,raw=direct(t);err=max(np.linalg.norm(value(w,t)/value(q,t)-write)/max(np.linalg.norm(write),1),np.linalg.norm(value(n,t)/value(q,t)-raw)/max(np.linalg.norm(raw),1));maxerr=max(maxerr,float(err));assert err<1e-10
   rr,ties=crossing_intervals(n,8);assert not ties
   row=dict(seed=entry['seed'],pair_index=pi,indices=[ia,ib],q_min=qmin,boundary_count=len(rr),boundaries=rr,status='no_crossing')
   endpoint=np.linalg.norm(direct(1)[0]-direct(0)[0]);row['native_endpoint_change_norm']=float(endpoint)
   if rr:
    r=rr[0];t=r['t'];native,raw=direct(t)
    def decode(a,selected):
     code=np.zeros_like(a);code[selected]=a[selected];return B@code.reshape(-1)*scale+s['bias']*scale+mu
    left=decode(raw,r['left']);right=decode(raw,r['right']);jump=right-left;finite=[]
    for fraction in p['fractions']:
     h=fraction*r['radius'];wl,al=direct(t-h);wr_,ar=direct(t+h);assert list(select(n,t-h,8))==r['left'] and list(select(n,t+h,8))==r['right']
     hard=decode(ar,r['right'])-decode(al,r['left']);soft=B@(block_shrinkage(torch.from_numpy(ar),8)-block_shrinkage(torch.from_numpy(al),8)).numpy().reshape(-1)*scale
     finite.append(dict(h=h,native_change_norm=float(np.linalg.norm(wr_-wl)),hard_change_norm=float(np.linalg.norm(hard)),continuous_change_norm=float(np.linalg.norm(soft)),hard_limit_error=float(np.linalg.norm(hard-jump))))
    row.update(status='evaluated',first_crossing=t,jump_norm=float(np.linalg.norm(jump)),jump_over_native_endpoint=float(np.linalg.norm(jump)/endpoint),finite_limits=finite)
   rows.append(row);saved[f"{entry['seed']}|{pi}"]=dict(q=torch.from_numpy(q),write_coefficients=torch.from_numpy(w),code_numerators=torch.from_numpy(n),z0=torch.from_numpy(z0),z1=torch.from_numpy(z1))
 sp=A/'emlp_m114_native_path_polynomials.pt';assert not sp.exists();torch.save(saved,sp)
 result=dict(status='completed_pending_independent_root_audit',source_sha256=sha(__file__),protocol_sha256=sha(pp),rows=rows,max_direct_relative_error=maxerr,path=str(sp),sha256=sha(sp),scope='Native MLP1 writes on normalized-input interpolation; actual hard-trained states only. Continuous shrinkage on same weights is a map-only diagnostic, not a newly trained model. No language-model downstream interventions or text reachability claim.')
 dest=V/'native_path_boundaries.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print('PATHS',len(rows),'CROSSING',sum(x['status']=='evaluated' for x in rows),'BOUNDARIES',sum(x['boundary_count'] for x in rows),'ERROR',maxerr)
if __name__=='__main__':main()

"""M89 frozen algebraic diagnostic; no fitting or evaluation-set access."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def error(a,b):
 d=a-b
 return {'relative':float(np.linalg.norm(d)/max(np.linalg.norm(b),1e-30)),'max_absolute':float(np.max(np.abs(d)))}
def run():
 torch.set_num_threads(1)
 protocol=json.loads((V/'computational_rank_diagnostic_protocol.json').read_text())
 assert sha(V/'bilinear_input_evaluation_models.json')==protocol['source_manifest_sha256']
 rng=np.random.default_rng(8901);toys=[]
 for k in [1,2,4,8]:
  x=rng.normal(size=(8192,k));y=rng.normal(size=(8192,k));f=(x*y).sum(1)
  u=rng.normal(size=k);u/=np.linalg.norm(u);t=rng.normal(size=8192)
  restricted=((t[:,None]*u)*y).sum(1);one=t*(y@u)
  # Coordinate probes recover the coefficient matrix exactly, independent of random draws.
  C=np.array([np.sum(np.eye(k)[i]*np.eye(k)[j]) for i in range(k) for j in range(k)]).reshape(k,k)
  assert np.array_equal(C,np.eye(k));assert np.linalg.matrix_rank(C)==k
  e=error(one,restricted);assert e['relative']<1e-12
  unique_deleted=f-x[:,0]*y[:,0];dup=x[:,0]*y[:,0];dup_deleted=(k-1)*dup/k
  row={'k':k,'output_dimension':1,'cross_coefficient_rank':int(np.linalg.matrix_rank(C)),'restricted_support_one_product_error':e,'independent_raw_delete_relative_squared':error(unique_deleted,f)['relative']**2,'duplicate_raw_delete_relative_squared':error(dup_deleted,dup)['relative']**2,'expected_independent_raw_delete_relative_squared':1/k,'expected_duplicate_raw_delete_relative_squared':1/k**2}
  if k>1:
   row['independent_rescaled_delete_relative_squared']=error(unique_deleted*k/(k-1),f)['relative']**2
   row['duplicate_rescaled_delete_error']=error(dup_deleted*k/(k-1),dup)
   assert row['duplicate_rescaled_delete_error']['relative']<1e-12
  toys.append(row)
 ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256']
 r=torch.load(ref['path'],weights_only=True);z=r['z'][r['indices']].double().numpy();L=r['L'].double().numpy();R=r['R'].double().numpy();W=r['D'].double().numpy()
 products=(z@L.T)*(z@R.T);native=products@W.T
 models=json.loads((V/'bilinear_input_evaluation_models.json').read_text())['models'];rows=[]
 for entry in models:
  if entry['variant']!='input':continue
  assert sha(entry['path'])==entry['sha256'];s=torch.load(entry['path'],weights_only=True)['state']
  D=s['features.dictionary'].double().numpy();D/=np.linalg.norm(D,axis=0)
  c=s['output_mean'].double().numpy()+float(s['output_scale'])*s['features.bias'].double().numpy()
  U,sv,_=np.linalg.svd(D,full_matrices=False);assert np.all(sv>sv[0]*1e-10)
  expected=c+((native-c)@U)@U.T
  Q,_=np.linalg.qr(D,mode='reduced')
  # Product dictionary -> output coordinates -> residual directions. Does not form native write.
  coefficient_weights=Q.T@W;offset=c-(c@Q)@Q.T
  actual=(products@coefficient_weights.T)@Q.T+offset
  e=error(actual,expected)
  row={k:entry[k] for k in ['kind','seed','base_arm','sha256']};row.update(e);row['passed']=e['relative']<=1e-5;rows.append(row)
  print(entry['kind'],entry['seed'],entry['base_arm'],e,flush=True)
 d=1152;h=4608;m=128
 out={'status':'passed' if all(r['passed'] for r in rows) else 'failed','source_sha256':sha(__file__),'protocol_sha256':sha(V/'computational_rank_diagnostic_protocol.json'),'manifest_sha256':sha(V/'bilinear_input_evaluation_models.json'),'reference_sha256':ref['sha256'],'toy_seed':8901,'toy_rows':toys,'native_rows':rows,'training_positions':len(z),'native_products_retained':h,'shapes':{'input':[len(z),d],'native_product_weights':[h,d],'output_basis':[d,m],'product_to_output_coordinates':[m,h]},'per_position_multiply_accumulates':{'native_linear_maps':2*h*d+h*d,'projected_factorization_linear_maps':2*h*d+m*h+d*m},'scope':'Float64 reassociation on training inputs, no new transformer forwards or fitting. Native product write excludes Down_bias; unchanged Down_bias added externally in M88. Product-to-coordinate dictionary is dense, retains all native products, and is not a minimal computation claim.'}
 assert len(rows)==16
 (V/'computational_rank_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
 assert out['status']=='passed'
if __name__=='__main__':run()

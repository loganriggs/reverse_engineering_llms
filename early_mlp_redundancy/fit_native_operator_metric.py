"""M100 training-only operator metric PCA, three fixed metrics, no heldout selection."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,A,sha
from native_operator_metric import operator_metric,fit_metric_pca,jacobian

def main():
 pp=V/'native_operator_metric_protocol.json';p=json.loads(pp.read_text())
 for name,h in p['sources'].items():assert sha(V.parent/name)==h
 for name,h in p['references'].items():assert sha(V/name)==h
 ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];r=torch.load(ref['path'],weights_only=True);L,R,D=[r[n].double().numpy() for n in ['L','R','D']];X=r['z'][r['indices']].double().numpy();assert X.shape==(4092,1152)
 ir=json.loads((V/'input_training_inventory.json').read_text());assert sha(ir['path'])==ir['sha256'];inputs=torch.load(ir['path'],weights_only=True)['inputs'];targets=json.loads((V/'training_causal_coverage.json').read_text())['records'];directions=[];metadata=[]
 for t in targets:
  if t['layer']!=1 or t['region']=='identity' or t['status']!='evaluated':continue
  key=t['pair_id']+'|'+str(t['endpoint']);dk=t['pair_id']+'|'+str(1-t['endpoint']);zr=inputs[key]['input'].reshape(-1,1152).double().numpy();zd=inputs[dk]['input'].reshape(-1,1152).double().numpy()
  for pos,dp in zip(t['recipient_positions'],t['donor_positions']):directions.append(zd[dp]-zr[pos]);metadata.append(dict(key=key,region=t['region'],recipient_position=pos,donor_position=dp))
 Z=np.stack(directions);assert len(Z)==296;S=Z.T@Z/len(Z);metrics={'euclidean':np.eye(1152),'isotropic_operator':operator_metric(L,R,D),'training_direction_operator':operator_metric(L,R,D,S)};C=(X-X.mean(0)).T@(X-X.mean(0))/len(X);models={};rows=[]
 for name,Q in metrics.items():
  checks=[]
  if name!='euclidean':
   cov=np.eye(1152) if name=='isotropic_operator' else S
   for a,b in p['training_check_indices']:
    delta=X[a]-X[b];J=jacobian(L,R,D,delta);want=float(np.sum((J@cov)*J));got=float(delta@Q@delta);relative=abs(got-want)/max(abs(want),1e-300);assert relative<p['metric_relative_tolerance'];checks.append(relative)
  m=fit_metric_pca(X,Q,1152,relative_cutoff=p['relative_cutoff']);eig=m['eigenvalues'];total=float(eig.sum());capacity=[]
  for k in p['ranks']:
   E=m['encoder'][:,:k];B=m['decoder'][:,:k];residual=np.eye(1152)-E@B.T;measured=float(np.sum((residual.T@C@residual)*Q.T));tail=float(eig[k:].sum());relative=abs(measured-tail)/max(total,1e-300);assert relative<1e-8
   capacity.append(dict(rank=k,training_metric_residual_ratio=measured/total,eigen_tail_ratio=tail/total,residual_identity_relative_error=relative))
  models[name]={k:torch.from_numpy(x) if isinstance(x,np.ndarray) else x for k,x in m.items()};rows.append(dict(metric=name,metric_rank=m['metric_rank'],metric_norm=float(np.linalg.norm(Q)),training_operator_energy=total,explicit_J_relative_errors=checks,capacity=capacity));print('FIT',name,capacity,flush=True)
 path=A/'emlp_m100_operator_metric_fits.pt';assert not path.exists();torch.save(dict(models=models,metrics={n:torch.from_numpy(q) for n,q in metrics.items()},direction_covariance=torch.from_numpy(S),direction_rows=metadata,protocol_sha256=sha(pp)),path)
 out=dict(status='fit_completed_audit_pending',source_sha256=sha(__file__),protocol_sha256=sha(pp),path=str(path),sha256=sha(path),training_positions=len(X),training_direction_positions=len(Z),rows=rows,scope='Training-only factorization; no nativeTransformerforwards orheldoutevaluation. Weights used explicitly. Operatorreconstruction doesnotbyitselfdefineanintegrable function orsemanticunit.')
 f=V/'native_operator_metric_fits.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print('DONE',flush=True)
if __name__=='__main__':main()

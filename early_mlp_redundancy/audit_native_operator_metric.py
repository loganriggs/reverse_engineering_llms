"""M100 independent Torch selected-Jacobian metric and projection/cross-metric audit."""
import json
import torch
from fit_pca_encoder import V,sha

def main():
 torch.set_num_threads(4);fp=V/'native_operator_metric_fits.json';f=json.loads(fp.read_text());assert sha(f['path'])==f['sha256'];p=json.loads((V/'native_operator_metric_protocol.json').read_text());assert f['protocol_sha256']==sha(V/'native_operator_metric_protocol.json')
 for n,h in p['sources'].items():assert sha(V.parent/n)==h
 for n,h in p['references'].items():assert sha(V/n)==h
 s=torch.load(f['path'],weights_only=True);ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];r=torch.load(ref['path'],weights_only=True);X=r['z'][r['indices']].double();L,R,D=[r[n].double() for n in ['L','R','D']];mean=X.mean(0);C=(X-mean).T@(X-mean)/len(X);ir=json.loads((V/'input_training_inventory.json').read_text());assert sha(ir['path'])==ir['sha256'];inputs=torch.load(ir['path'],weights_only=True)['inputs'];directions=[]
 expected=[]
 for t in json.loads((V/'training_causal_coverage.json').read_text())['records']:
  if t['layer']==1 and t['region']!='identity' and t['status']=='evaluated':
   for pos,dp in zip(t['recipient_positions'],t['donor_positions']):expected.append(dict(key=t['pair_id']+'|'+str(t['endpoint']),region=t['region'],recipient_position=pos,donor_position=dp))
 assert s['direction_rows']==expected and len(expected)==296
 training_ids={x['key'].rsplit('|',1)[0] for x in expected};assert len(training_ids)==70;assert not training_ids&set(json.loads((V/'diverse_supervision_split.json').read_text())['evaluation_pair_ids'])
 for t in expected:
  pid,ep=t['key'].rsplit('|',1);a=inputs[t['key']]['input'].reshape(-1,1152)[t['recipient_position']].double();b=inputs[pid+'|'+str(1-int(ep))]['input'].reshape(-1,1152)[t['donor_position']].double();directions.append(b-a)
 Z=torch.stack(directions);S=Z.T@Z/len(Z);assert torch.allclose(S,s['direction_covariance'],rtol=1e-12,atol=1e-12)
 indices=[0,127,511,1151];Js=[(D*R[:,i])@L+(D*L[:,i])@R for i in indices];metric_checks=[]
 for name,cov in [('isotropic_operator',torch.eye(1152,dtype=torch.float64)),('training_direction_operator',S)]:
  explicit=torch.stack([torch.stack([((a@cov)*b).sum() for b in Js]) for a in Js]);stored=s['metrics'][name][indices][:,indices];error=float(torch.linalg.norm(explicit-stored)/torch.linalg.norm(explicit));assert error<1e-9;metric_checks.append(dict(metric=name,selected_indices=indices,relative_error=error))
 cross=[]
 for name,m in s['models'].items():
  assert torch.allclose(m['mean'],mean,atol=1e-12,rtol=1e-12);Q=s['metrics'][name]
  for k in p['ranks']:
   E=m['encoder'][:,:k];B=m['decoder'][:,:k];P=E@B.T;ident=torch.eye(1152,dtype=torch.float64);res=ident-P;rcov=res.T@C@res
   assert torch.allclose(B.T@E,torch.eye(k,dtype=torch.float64),atol=1e-8,rtol=1e-8)
   idem=float(torch.linalg.norm(P@P-P)/torch.linalg.norm(P));adj=float(torch.linalg.norm(P@Q-Q@P.T)/torch.linalg.norm(Q));assert idem<1e-8 and adj<1e-8
   for target,T in s['metrics'].items():
    err=float((rcov*T.T).sum());total=float((C*T.T).sum());ratio=err/total;cross.append(dict(basis=name,rank=k,measured_metric=target,relative_residual=ratio,idempotence_relative_error=idem,metric_adjoint_relative_error=adj))
    if target==name:
     reported=next(x for x in f['rows'] if x['metric']==name);row=next(x for x in reported['capacity'] if x['rank']==k);assert abs(ratio-row['training_metric_residual_ratio'])<1e-9
 # In each matching objective its own metric-PCA must beat other same-rank bases.
 for target in s['metrics']:
  for k in p['ranks']:
   rr=[x for x in cross if x['rank']==k and x['measured_metric']==target];own=next(x['relative_residual'] for x in rr if x['basis']==target);assert all(own<=x['relative_residual']+1e-9 for x in rr)
 out=dict(status='passed',source_sha256=sha(__file__),fit_sha256=sha(fp),metric_checks=metric_checks,direction_positions=296,training_pairs=70,cross_metric_rows=cross,scope='Independent Torch covariance, explicit selectedJacobian Gram entries, mean/projectionidempotence/Q-adjoint checks, sameobjective fit replay and27crossmetric comparisons. No independent full-Q or full eigensolver recomputation; no heldout inference.')
 dest=V/'native_operator_metric_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('PASSED',metric_checks);print(json.dumps(cross,indent=2))
if __name__=='__main__':main()

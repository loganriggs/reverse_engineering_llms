"""Independent NumPy gradient accounting; not independent transformer backprop."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rp=P/'family_gradient_compatibility.json';r=json.loads(rp.read_text());pp=P/'family_gradient_protocol.json';p=json.loads(pp.read_text());assert r['protocol_sha256']==sha(pp);assert p['initial_protocol_sha256']==sha(P/'response_supervision_protocol.json');assert p['targets_sha256']==r['targets_sha256']==sha(P/'response_training_targets.json');sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];saved=torch.load(sp,weights_only=True);assert len(saved)==len(r['models'])==4;summary=[];maxerr=0
for m,s in zip(r['models'],saved):
 assert (m['kind'],m['seed'])==(s['kind'],s['seed']);assert m['names']==s['names'];G=s['gradients'].numpy();names=m['names'];g={n:G[i] for i,n in enumerate(names)};gram=G@G.T;maxerr=max(maxerr,float(np.abs(gram-np.array(m['gram'])).max()));np.testing.assert_allclose(gram,m['gram'],rtol=1e-9,atol=1e-12);norm=np.sqrt(np.diag(gram));np.testing.assert_allclose(norm,m['norms'],rtol=1e-9,atol=1e-12);den=norm[:,None]*norm[None,:];cos=np.divide(gram,den,out=np.zeros_like(gram),where=den>0);np.testing.assert_allclose(cos,m['cosine'],rtol=1e-9,atol=1e-12)
 families=list(m['family_endpoint_counts']);assert len(families)==10 and sum(m['family_endpoint_counts'].values())==120
 for fam in families:np.testing.assert_allclose(g[fam+':combined'],g[fam+':K']+m['weight']*g[fam+':R'],rtol=1e-9,atol=1e-12)
 for key,suffix,w in [('task:K','K',1),('task:weighted_R','R',m['weight'])]:np.testing.assert_allclose(g[key],sum(g[f+':'+suffix]*n*w/120 for f,n in m['family_endpoint_counts'].items()),rtol=1e-9,atol=1e-12)
 np.testing.assert_allclose(g['schedule:combined'],(g['natural:K']+g['task:K']+g['task:weighted_R'])/2,rtol=1e-9,atol=1e-12)
 for check in m['finite_difference_checks']:assert min(x['relative_error'] for x in check['finite_differences'])<.03
 def cosine(a,b):return float(cos[names.index(a),names.index(b)])
 pairs=[cosine(a+':R',b+':R') for i,a in enumerate(families) for b in families[i+1:]]
 row=dict(kind=m['kind'],seed=m['seed'],negative_response_family_pairs=sum(x<0 for x in pairs),total_family_pairs=len(pairs),task_K_vs_weighted_R=cosine('task:K','task:weighted_R'),natural_K_vs_weighted_R=cosine('natural:K','task:weighted_R'),natural_K_vs_task_K=cosine('natural:K','task:K'),weighted_R_norm=m['norms'][names.index('task:weighted_R')],task_K_norm=m['norms'][names.index('task:K')],natural_K_norm=m['norms'][names.index('natural:K')],families_harmed_by_aggregate_descent=[f for f in families if cosine(f+':R','schedule:combined')<0],family_response_aggregate_cosines={f:cosine(f+':R','schedule:combined') for f in families})
 summary.append(row);print(json.dumps(row))
o=dict(status='passed',result_sha256=sha(rp),models=4,max_gram_absolute_error=maxerr,summary=summary,scope='NumPy Gram/norm/cosine and all family/schedule aggregation checks; parameter finite differences on one fixed endpoint per model. Does not independently reproduce backpropagation or prove global feasibility. Negative gradient dot means infinitesimal descent on one increases the other at these starts.')
(P/'family_gradient_audit.json').write_text(json.dumps(o,indent=2)+'\n')

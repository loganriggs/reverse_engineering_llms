"""M102 training-only variance context for the rank-matched random null."""
import json
import torch
from fit_pca_encoder import V,sha

def main():
 torch.set_num_threads(1);rf=V/'native_random_subspaces.json';r=json.loads(rf.read_text());a=json.loads((V/'native_random_subspaces_audit.json').read_text());assert a['status']=='passed' and a['fit_sha256']==sha(rf);assert sha(r['path'])==r['sha256'];random=torch.load(r['path'],weights_only=True)['models'];pf=V/'native_operator_metric_fits.json';p=json.loads(pf.read_text());assert sha(p['path'])==p['sha256'];data=torch.load(p['path'],weights_only=True)['models']['euclidean'];ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];s=torch.load(ref['path'],weights_only=True);X=s['z'][s['indices']].double()-s['mean'];total=float(X.square().sum());rows=[]
 for basis,m in dict(data_PCA=data,**random).items():
  for k in [4,16,64]:
   E=m['encoder'][:,:k];B=m['decoder'][:,:k];assert torch.equal(E,B);code=X@E;captured=float(code.square().sum());residual=float((X-code@B.T).square().sum());assert abs(captured+residual-total)<1e-10*total;rows.append(dict(basis=basis,rank=k,captured_fraction=captured/total,residual_fraction=residual/total,isotropic_random_expected_fraction=k/1152))
 out=dict(status='completed',source_sha256=sha(__file__),random_fit_sha256=sha(rf),data_fit_sha256=sha(pf),rows=rows,scope='Training-only centeredinputvariance; Euclideanorthogonalprojections, directPythagoreancheck. Rankmatchedisnotvariance-matched. No fit/selection orheldoutoutcomes used.')
 f=V/'native_random_training_variance.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()

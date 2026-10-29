"""M102 frozen Gaussian/QR subspaces, independent of any activation or task outcome."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,A,sha

def main():
 dp=V/'native_mixed_random_design.json';d=json.loads(dp.read_text())
 for name,h in d['references'].items():assert sha(V/name)==h
 orientations={}
 for seed in d['seeds']:
  gaussian=np.random.default_rng(seed).normal(size=(1152,64));Q,R=np.linalg.qr(gaussian,mode='reduced');Q=Q*np.where(np.diag(R)<0,-1.,1.);orientations[seed]=Q
 # Only after generating orientations, load the fixed shared mean.
 ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];r=torch.load(ref['path'],weights_only=True);models={}
 for seed,Q in orientations.items():models['random_'+str(seed)]=dict(mean=r['mean'],encoder=torch.from_numpy(Q),decoder=torch.from_numpy(Q.copy()),random_seed=seed)
 path=A/'emlp_m102_random_subspaces.pt';assert not path.exists();torch.save(dict(models=models,design_sha256=sha(dp),reference_sha256=sha(V/'native_input_reference.json')),path)
 out=dict(status='prepared_audit_pending',source_sha256=sha(__file__),design_sha256=sha(dp),path=str(path),sha256=sha(path),seeds=d['seeds'],ranks=d['ranks'],scope='Gaussian orientations generatedbeforeloadingnativeinputreference; sharedmeanfixedfromtraining. Allseeds/ranks retained; nofitting/selection. Nestedprefixsubspaces.')
 f=V/'native_random_subspaces.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print('PREPARED',len(models))
if __name__=='__main__':main()

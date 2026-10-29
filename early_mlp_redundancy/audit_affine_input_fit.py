"""M78 independent Torch QR/SVD solve and empirical optimality checks."""
import json
import numpy as np
import torch
from fit_pca_encoder import sha,V
P=V.parent

def main():
    torch.set_num_threads(1);fit=json.loads((V/'affine_input_fit.json').read_text());assert fit['source_sha256']==sha(P/'fit_affine_input_predictor.py') and sha(fit['path'])==fit['sha256'];execution=json.loads((V/'affine_input_execution.json').read_text());assert fit['execution_sha256']==sha(V/'affine_input_execution.json');s=np.load(fit['path'])
    ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];z=torch.load(ref['path'],weights_only=True)['z'];cache=json.loads((V/'natural_token_context.json').read_text());assert sha(cache['cache'])==cache['cache_sha256'];d=torch.load(cache['cache'],weights_only=True)
    indices=torch.tensor([int(i) for i in d['unique_indices'] if int(d['documents'][i])<32]);assert np.array_equal(indices.numpy(),s['indices']);x=z[indices].double();y=d['Y'][1][indices].double();mx=sum(t.sum(0) for t in x.split(117))/len(x);my=sum(t.sum(0) for t in y.split(117))/len(y);assert np.allclose(mx.numpy(),s['mean_input'],rtol=1e-12,atol=1e-12) and np.allclose(my.numpy(),s['mean_write'],rtol=1e-12,atol=1e-12)
    x=x-mx;y=y-my;q,r=torch.linalg.qr(x);u,sv,vh=torch.linalg.svd(r,full_matrices=False);keep=sv>execution['relative_singular_cutoff']*sv[0];assert int(keep.sum())==fit['rank'];coef=(vh[keep].T/sv[keep])@(u[:,keep].T@(q.T@y));saved=torch.from_numpy(s['coefficient']);relative=float(torch.linalg.norm(coef-saved)/torch.linalg.norm(saved));assert relative<execution['independent_coefficient_relative_tolerance']
    residual=x@saved-y;normal=float(torch.linalg.norm(x.T@residual)/(torch.linalg.norm(x)*torch.linalg.norm(residual)));assert normal<execution['normal_equation_relative_tolerance']
    projected=q@(u[:,keep]@(u[:,keep].T@(q.T@y)));gap=float(torch.linalg.norm(x@saved-projected)/torch.linalg.norm(projected));assert gap<execution['training_projection_relative_tolerance'];energy=float(residual.square().sum());assert np.isclose(energy,fit['train_residual_energy'],rtol=1e-10)
    out=dict(status='passed',source_sha256=sha(__file__),fit_sha256=sha(V/'affine_input_fit.json'),coefficient_relative_difference=relative,normal_equation_relative_residual=normal,training_projection_relative_error=gap,rank=int(keep.sum()),scope='Independent chunked means and index selection, Torch QR then small-matrix SVD, same numerical rank, normal equations and projected optimum verified. No heldout scoring.')
    (V/'affine_input_fit_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()

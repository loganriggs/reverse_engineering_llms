"""M78 one frozen empirical affine input-to-write least-squares fit."""
import json
import numpy as np
import torch
from fit_pca_encoder import sha,V,A
P=V.parent

def main():
    torch.set_num_threads(1);pp=V/'affine_input_predictor_protocol.json';protocol=json.loads(pp.read_text())
    for name,digest in protocol['references'].items():assert sha(V/name)==digest
    execution=dict(protocol_sha256=sha(pp),relative_singular_cutoff=1e-10,independent_coefficient_relative_tolerance=1e-7,normal_equation_relative_tolerance=1e-10,training_projection_relative_tolerance=1e-9,forward_override_max_logit_difference=1e-5,scope='One centered float64 SVD fit; no regularization or rank choice.')
    ep=V/'affine_input_execution.json';assert not ep.exists();ep.write_text(json.dumps(execution,indent=2)+'\n')
    ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];state=torch.load(ref['path'],weights_only=True)
    cache=json.loads((V/'natural_token_context.json').read_text());assert sha(cache['cache'])==cache['cache_sha256'];data=torch.load(cache['cache'],weights_only=True)
    indices=data['unique_indices'][data['documents'][data['unique_indices']]<32];assert torch.equal(indices,state['indices']) and len(indices)==4092
    cohort=json.loads((V/'natural_token_context_cohort.json').read_text());assert {x['document_hash'] for x in cohort['records'][:32]}.isdisjoint({x['document_hash'] for x in cohort['records'][32:]})
    x=state['z'][indices].double().numpy();y=data['Y'][1][indices].double().numpy();mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my
    u,sv,vt=np.linalg.svd(xc,full_matrices=False);keep=sv>execution['relative_singular_cutoff']*sv[0];coef=(vt[keep].T/sv[keep])@(u[:,keep].T@yc);residual=xc@coef-yc
    path=A/'emlp_m78_affine_state.npz';assert not path.exists();np.savez(path,mean_input=mx,mean_write=my,coefficient=coef,singular_values=sv,indices=indices.numpy(),retained=keep)
    result=dict(status='fitted_pending_audit',source_sha256=sha(__file__),execution_sha256=sha(ep),path=str(path),sha256=sha(path),positions=len(indices),rank=int(keep.sum()),condition_number=float(sv[0]/sv[keep][-1]),coefficient_frobenius_norm=float(np.linalg.norm(coef)),effective_affine_parameters=1152*1152+1152,stored_float64_scalars=int(coef.size+mx.size+my.size),train_relative_raw_error=float(np.square(residual).sum()/np.square(y).sum()),train_relative_centered_error=float(np.square(residual).sum()/np.square(yc).sum()),train_residual_energy=float(np.square(residual).sum()))
    (V/'affine_input_fit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()

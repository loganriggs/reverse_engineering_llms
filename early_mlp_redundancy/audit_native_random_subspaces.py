"""M102 deterministic replay and independent Torch QR/projection verification."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha

def main():
 torch.set_num_threads(1);fp=V/'native_random_subspaces.json';f=json.loads(fp.read_text());assert f['source_sha256']==sha(V.parent/'prepare_native_random_subspaces.py');assert sha(f['path'])==f['sha256'];s=torch.load(f['path'],weights_only=True);d=json.loads((V/'native_mixed_random_design.json').read_text());assert f['design_sha256']==sha(V/'native_mixed_random_design.json');ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];r=torch.load(ref['path'],weights_only=True);rows=[]
 for seed in d['seeds']:
  m=s['models']['random_'+str(seed)];assert torch.equal(m['mean'],r['mean']) and torch.equal(m['encoder'],m['decoder']);G=np.random.default_rng(seed).normal(size=(1152,64));qn,rn=np.linalg.qr(G,mode='reduced');qn*=np.where(np.diag(rn)<0,-1.,1.);assert np.array_equal(qn,m['encoder'].numpy());qt,rt=torch.linalg.qr(torch.from_numpy(G),mode='reduced');qt*=torch.where(torch.diag(rt)<0,-1.,1.)
  for k in d['ranks']:
   Q=m['encoder'][:,:k];P=Q@Q.T;orth=float(torch.linalg.norm(Q.T@Q-torch.eye(k)));idem=float(torch.linalg.norm(P@P-P)/torch.linalg.norm(P));ind=float(torch.linalg.norm(P-qt[:,:k]@qt[:,:k].T)/torch.linalg.norm(P));assert orth<1e-12 and idem<1e-12 and ind<1e-12;rows.append(dict(seed=seed,rank=k,orthogonality_error=orth,idempotence_relative_error=idem,independent_QR_projector_error=ind))
 train=json.loads((V/'natural_token_context_cohort.json').read_text());assert {x['document_hash'] for x in train['records'][:32]}.isdisjoint({x['document_hash'] for x in train['records'][32:]})
 out=dict(status='passed',source_sha256=sha(__file__),fit_sha256=sha(fp),rows=rows,scope='ExactNumPy RNG/QRreplay; independentTorchQR projector, orthogonality/idempotence, nestedprefixranks/sharedmean, naturaldocumentsplit. Orientations not selectedfromdata oroutcomes. Noheldoutperformance.')
 p=V/'native_random_subspaces_audit.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print('AUDITED',len(rows))
if __name__=='__main__':main()

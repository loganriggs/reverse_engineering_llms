"""M72 all-endpoint preflight: LS optimality, independent QR solve and affine delta."""
from pathlib import Path
import json,torch,numpy as np
from train_write_featurizers import Featurizer
from least_squares_encoder import LeastSquaresEncoder
from prepare_encoded_span_diagnostic import sha
P=Path(__file__).resolve().parent;V=P/'validation'
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);pp=V/'least_squares_encoder_protocol.json';q=json.loads(pp.read_text());rp=V/'encoded_factorial.json';mp=V/'encoded_change_evaluation_models.json';assert sha(rp)==q['factorial_sha256'] and sha(mp)==q['models_sha256'];old=json.loads(rp.read_text());models=json.loads(mp.read_text())['models'];n=0;maxqr=0.;maxorth=0.;maxdelta=0.;minrank=128;maxrank=0;states={};checks=[]
 for e in models:
  path=Path(e['path']);assert sha(path)==e['sha256'];s=torch.load(path,weights_only=True);f=Featurizer(e['kind'],e['seed']).cuda();f.load_state_dict(s['state']);states[tuple(e[k] for k in ['kind','seed','arm'])]=LeastSquaresEncoder(f,s['mean'].cuda(),s['scale'].cuda())
 for file in old['files']:
  p=Path(file['path']);assert sha(p)==file['sha256'];b=torch.load(p,weights_only=True)
  for e in models:
   mk=tuple(e[k] for k in ['kind','seed','arm']);ms=':'.join(map(str,mk));enc=states[mk];dense=[]
   for ep in [0,1]:
    w=b['writes'][ep].cuda();original,oq,oc,_=enc.reconstruct(w,'original');assert torch.equal(original.cpu(),b['reconstructions'][ms+f'|{ep}']['write']) and torch.equal(oc.cpu(),b['reconstructions'][ms+f'|{ep}']['code']);target=((w.cpu().double().numpy()-enc.mu)/enc.s-enc.bias).reshape(-1,1152);prev_error=float((original.double()-w.double()).square().sum())
    for variant in ['own_support_LS','full_dictionary_LS']:
     rec,support,code,info=enc.reconstruct(w,variant);assert np.isfinite(rec.cpu().numpy()).all();inds=support.cpu().numpy().reshape(-1,128)>0;cn=code.cpu().numpy().reshape(-1,128)
     if variant=='own_support_LS':assert torch.equal(support,oq)
     else:assert bool((support==1).all())
     for i,mask in enumerate(inds):
      D=enc.D[:,mask];qmat,rmat=np.linalg.qr(D,mode='reduced');u,sv,_=np.linalg.svd(rmat,full_matrices=False);Q=qmat@u[:,sv>1e-10*sv[0]];expected=Q@(Q.T@target[i]);got=cn[i]@enc.D.T;rel=np.linalg.norm(expected-got)/max(np.linalg.norm(expected),1.);orth=np.linalg.norm(D.T@(target[i]-got))/max(np.linalg.norm(target[i]),1.);maxqr=max(maxqr,float(rel));maxorth=max(maxorth,float(orth));assert rel<1e-9 and orth<1e-9;minrank=min(minrank,Q.shape[1]);maxrank=max(maxrank,Q.shape[1])
     error=float((rec.double()-w.double()).square().sum());tol=1e-6*max(float(w.double().square().sum()),1.);assert error<=prev_error+tol;prev_error=error;n+=1
     if variant=='full_dictionary_LS':dense.append(rec.cpu().double().numpy())
    checks.append(dict(kind=e['kind'],seed=e['seed'],arm=e['arm'],key=file['pair_id']+f'|{ep}',original_support_mean=float(oq.sum(-1).float().mean()),full_rank_mean=info['rank_mean']))
   # Full per-input differences versus fixed linear projection, including cast rounding.
   pinv,_=enc.solver(tuple(range(128)));a=b['writes'][0].double().numpy();d=b['writes'][1].double().numpy();assert a.shape==d.shape;projected=((d-a)@pinv.T)@enc.D.T;rel=float(np.square((dense[1]-dense[0])-projected).sum())/max(float(np.square(projected).sum()),1.);maxdelta=max(maxdelta,rel);assert rel<1e-8
  print('PREFLIGHT_PAIRS',old['files'].index(file)+1,flush=True)
 assert n==1008 and len(checks)==504
 op=V/'least_squares_encoder_preflight.json';assert not op.exists();op.write_text(json.dumps(dict(status='passed',source_sha256=sha(Path(__file__)),encoder_source_sha256=sha(P/'least_squares_encoder.py'),protocol_sha256=sha(pp),models_sha256=sha(mp),reconstructions=n,original_replays=504,checks=checks,max_QR_relative_error=maxqr,max_orthogonality_error=maxorth,max_dense_difference_relative_squared_error=maxdelta,min_rank=minrank,max_rank=maxrank,scope='All fixed task endpoints independently encoded; own supports replay original, separate QR projections verify LS; local optimality and dense affine differences including float32 casts. No native causal or natural retention results yet.'),indent=2)+'\n');print('PASS',n,maxqr,maxdelta,flush=True)
if __name__=='__main__':main()

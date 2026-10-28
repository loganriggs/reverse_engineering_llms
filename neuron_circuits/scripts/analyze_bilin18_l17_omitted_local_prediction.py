from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_omitted_local_prediction';j=json.loads((A/f'{stem}.json').read_text());old=np.load(A/'bilin18_l17_baseline_context_confirmation_rows.npz');z=np.load(A/f'{stem}_rows.npz');err=z['errors'];base=old['errors'][0,:,:,:,2];effects=old['effects'][0];assert err.shape==(3,12,32,32,2) and np.isfinite(err).all()
def matched(src,ix,metric):
 ee=err[:,src][:,:,ix,ix,metric].sum((1,2));bb=base[src][:,ix,ix,metric].sum();ff=effects[src][:,ix,ix,metric].sum();return 1-ee/bb,1-ee/ff
src=np.arange(12);lo=np.arange(16);mi=np.arange(16,32);point=np.stack([matched(src,ix,0)[0][:2] for ix in [lo,mi]]).T.reshape(-1);rng=np.random.default_rng(j['config']['bootstrap_seed']);boot=[]
for _ in range(j['config']['bootstrap_samples']):
 pairs=rng.integers(0,6,6);ss=np.r_[pairs,6+pairs];ll=rng.integers(0,16,16);mm=16+rng.integers(0,16,16);boot.append(np.stack([matched(ss,ix,0)[0][:2] for ix in [ll,mm]]).T.reshape(-1))
boot=np.array(boot);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],1);j.update(primary_names=['linear_low','linear_median','second_order_low','second_order_median'],primary_omitted_Fisher_recovery=point.tolist(),primary_joint4_ci95=ci.tolist(),above90=(ci[:,0]>.90).tolist(),matched={name:{metric:{'omitted_error_recovery':matched(src,ix,mi)[0].tolist(),'full_effect_recovery':matched(src,ix,mi)[1].tolist()} for metric,mi in [('Fisher',0),('KL',1)]} for name,ix in [('low',lo),('median',mi)]})
# Independent central-difference audit of normalized-input second-order coefficient.
p=old['pre'].astype(float)[None];v=-old['sources'].astype(float)[:,None];eps=np.finfo(np.float32).eps;s=np.sqrt(np.mean(p*p,-1,keepdims=True)+eps);m=np.mean(p*v,-1,keepdims=True);v2=np.mean(v*v,-1,keepdims=True);u2=p*(-v2/(2*s**3)+3*m*m/(2*s**5))-v*m/s**3
norm=lambda y:y/np.sqrt(np.mean(y*y,-1,keepdims=True)+eps)
t=.001;fd=(norm(p+t*v)+norm(p-t*v)-2*norm(p))/(2*t*t);relative=np.linalg.norm(fd-u2,axis=-1)/np.maximum(np.linalg.norm(u2,axis=-1),1e-12);j['RMS_second_order_FD_relative_max']=float(relative.max());j['RMS_second_order_FD_note']='Independent central difference at step0.001, descriptive numerical audit; no statistical gate is changed.'
np.savez_compressed(A/f'{stem}_bootstrap.npz',primary=boot);(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps({k:j[k] for k in ['primary_omitted_Fisher_recovery','primary_joint4_ci95','above90','matched','RMS_second_order_FD_relative_max']},indent=2))

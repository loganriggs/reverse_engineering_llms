from pathlib import Path
import numpy as np,json
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';s='bilin18_l17_canonical_width_budget';audits={}
# Kernel queries independent of fitting: all128 held-out normalized test inputs.
pp=np.empty((512,1152),np.float32)
for p in ['compact_operation_confirmation','unprojected_finite_confirmation','covariance_source_confirmation','covariance_fresh_confirmation']:
 p='bilin18_l17_'+p;ids=json.loads((A/f'{p}_selection.json').read_text())['sites'];pp[ids]=np.load(A/f'{p}_rows.npz')['pre']
te=json.loads((A/'bilin18_l17_data_canonical_pilot_selection.json').read_text())['test'];x=pp[te].astype(np.float64);x/=np.sqrt((x*x).mean(1,keepdims=True)+np.finfo(np.float32).eps)
for rank in [64,128]:
 z=np.load(A/f'{s}_rank{rank}_models.npz');support=z['support'];E=z['kernel_encoder'];km=z['kernel_mean'];grand=float(z['kernel_grand']);k=(x@support.T/1152)**2;out={'support':support,'scale':z['scale']};audits[str(rank)]={}
 for name in ['initial','budget8','budget32']:
  C=z[name+'_map'].astype(np.float64);b=z[name+'_intercept'].astype(np.float64);u,sv,vh=np.linalg.svd(C,full_matrices=False);read=E@(u[:,:16]*sv[:16]);write=vh[:16];rowmean=read.sum(0);intercept=b+(grand-km)@read@write
  original=(k-km[None]-k.mean(1,keepdims=True)+grand)@E@C+b;compressed=(k@read-k.mean(1,keepdims=True)*rowmean[None])@write+intercept;err=float(np.linalg.norm(original-compressed)/np.linalg.norm(original));assert err<1e-6
  out[name+'_read']=read;out[name+'_write']=write;out[name+'_rowmean_read']=rowmean;out[name+'_intercept']=intercept;audits[str(rank)][name]={'relative_prediction_error':err,'rank16_truncation_relative':float(np.sqrt((sv[16:]**2).sum()/(sv**2).sum())),'per_map_scalars_excluding_shared_support_scale':int(read.size+write.size+rowmean.size+intercept.size)}
 np.savez_compressed(A/f'{s}_rank{rank}_compiled.npz',**out)
result={'scope':'Exact algebra before floating-point rank truncation; held-out query check in float64. No outcome fitting. Same support dictionary and rank16 final read/write sizes across both training widths. No runtime speed benchmark.','shared_support_scalars':int(support.size),'shared_scale_scalars':1,'audits':audits};(A/f'{s}_compilation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

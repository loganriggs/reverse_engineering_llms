from pathlib import Path
import numpy as np,json
A=Path(__file__).resolve().parents[1]/'artifacts/bilin18_tensor_decomposition';maps=np.load(A/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['maps'];B=np.load(A/'bilin18_l17_activity_compiled_maps_frozen_basis.npz')['basis'];pred=np.load(A/'bilin18_l17_activity_compiled_maps_rows.npz')['predictions'];result={}
for label,stem in [('calibration','bilin18_l17_baseline_context_confirmation'),('transfer','bilin18_l17_operator_transfer_confirmation')]:
 p=np.load(A/f'{stem}_rows.npz')['pre'].astype(float);x=p/np.sqrt(np.mean(p*p,-1,keepdims=True)+np.finfo(np.float32).eps);act=np.einsum('roi,ni->nro',maps,x,optimize=True);flat=act.reshape(-1,1152);rows={}
 for rank in [16,32,64,128,256]:
  b=B[:,:rank];proj=(flat@b)@b.T;rows[str(rank)]={'pooled_activity_recovery':float(1-np.sum((flat-proj)**2)/np.sum(flat**2))}
 result[label]=rows
for ri,rank in enumerate([16,32,64,128,256]):
 for label,ix in [('low',np.arange(16)),('median',np.arange(16,32))]:
  exact=pred[0,:,ix];miss=pred[2+2*ri,:,ix]-exact;result['transfer'][str(rank)][label+'_relative_raw_correction_error']=float(np.sum(miss**2)/np.sum(exact**2))
p=A/'bilin18_l17_activity_compiled_maps.json';j=json.loads(p.read_text());j['post_primary_activity_transfer_diagnostic']=result;p.write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(result,indent=2))

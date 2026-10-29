"""M87 freeze all32final evaluation candidates and verify information access."""
import json
import numpy as np
import torch
from bilinear_input_evaluation import BilinearEvaluationPredictor
from fit_pca_encoder import sha,V
P=V.parent

def main():
 torch.set_num_threads(1);pp=V/'bilinear_input_evaluation_preflight_protocol.json';tol=json.loads(pp.read_text())['tolerances'];rp=V/'bilinear_input_joint_training_results.json';result=json.loads(rp.read_text());audit=json.loads((V/'bilinear_input_final_loss_audit.json').read_text());assert audit['status']=='passed' and audit['results_sha256']==sha(rp);ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];raw=torch.load(ref['path'],weights_only=True)['z'][:128].cuda()[None];m4=json.loads((V/'natural_token_context.json').read_text());assert sha(m4['cache'])==m4['cache_sha256'];write=torch.load(m4['cache'],weights_only=True)['Y'][1][:128].cuda()[None];models=[];checks=[]
 for e in result['models']:
  assert sha(e['path'])==e['sha256'];s=torch.load(e['path'],weights_only=True)['state'];d={k:v.double().numpy() for k,v in s.items()};x=(raw.cpu().double().numpy()-d['input_mean'])/d['input_scale'];a=(x@d['features.left'].T+d['features.left_bias'])*(x@d['features.right'].T+d['features.right_bias']);M=d['features.membership'];support=np.ones_like(a)
  if e['kind']!='dense':
   scores=np.abs(a) if e['kind']=='scalar' else a*a@M.T;k=32 if e['kind']=='scalar' else (8 if e['kind']=='block' else 4);ids=np.argsort(-scores,axis=-1,kind='stable')[...,:k];selected=np.zeros_like(scores);np.put_along_axis(selected,ids,1,axis=-1);support=selected if e['kind']=='scalar' else (selected@M>0)
  D=d['features.dictionary']/np.linalg.norm(d['features.dictionary'],axis=0);center=d['output_mean']+d['output_scale']*d['features.bias'];reference=(a*support)@D.T*d['output_scale']+center
  for variant in ['input','output_oracle']:
   entry=e|dict(base_arm=e['arm'],arm=e['arm']+'__'+variant,variant=variant);enc=BilinearEvaluationPredictor(entry);q,active,code,info=enc.reconstruct(raw,write);actual=q.cpu().double().numpy()
   if variant=='input':
    error=float(np.linalg.norm(actual-reference)/max(1,np.linalg.norm(reference)));assert error<tol['input_relative'];assert np.array_equal(active.cpu().numpy(),support);orth=None
   else:
    y=write.cpu().double().numpy()-center;coef=np.linalg.lstsq(D,y.reshape(-1,1152).T,rcond=1e-10)[0];independent=(D@coef).T.reshape(y.shape)+center;error=float(np.linalg.norm(actual-independent)/max(1,np.linalg.norm(independent)));assert error<tol['oracle_relative'];residual=y.reshape(-1,1152).T-D@coef;orth=float(np.linalg.norm(D.T@residual)/max(1,np.linalg.norm(residual)));assert orth<tol['orthogonality_relative']
    try:enc.reconstruct(raw)
    except ValueError:pass
    else:raise AssertionError('Oracle accepted input without native write')
   models.append(entry);checks.append(dict(kind=e['kind'],seed=e['seed'],arm=e['arm'],variant=variant,relative_prediction_error=error,residual_orthogonality=orth))
 manifest=dict(models=models,training_results_sha256=sha(rp),final_loss_audit_sha256=sha(V/'bilinear_input_final_loss_audit.json'),prospective_evaluation_sha256=sha(V/'bilinear_input_prospective_evaluation.json'),encoder_source_sha256=sha(P/'bilinear_input_evaluation.py'),scope='All16finalmodels and16privilegedoutputprojections. No selection; inputpredictors andnativewriteoracles distinguished byvariant.');op=V/'bilinear_input_evaluation_models.json';assert not op.exists();op.write_text(json.dumps(manifest,indent=2)+'\n');out=dict(status='passed',source_sha256=sha(__file__),protocol_sha256=sha(pp),manifest_sha256=sha(op),checks=checks,training_document=0,native_forwards=0,scope='Independent localarithmetic andprojection tests only; noheldoutperformance.');dest=V/'bilinear_input_evaluation_preflight.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('PASSED',len(models),'candidates')
if __name__=='__main__':main()

"""Freeze training-only normalization and all eight untrained initial states."""
import json
import numpy as np
import torch
from bilinear_input_featurizer import BilinearInputFeaturizer,KINDS
from fit_pca_encoder import sha,V,A
P=V.parent

def main():
 torch.set_num_threads(1);protocol=V/'bilinear_input_featurizer_preflight_protocol.json';q=json.loads(protocol.read_text())
 for name,digest in q['references'].items():assert sha(V/name)==digest
 checks=json.loads((V/'bilinear_input_architecture_checks.json').read_text());assert checks['status']=='passed' and checks['module_sha256']==sha(P/'bilinear_input_featurizer.py')
 ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];s=torch.load(ref['path'],weights_only=True)
 m4=json.loads((V/'natural_token_context.json').read_text());assert sha(m4['cache'])==m4['cache_sha256'];data=torch.load(m4['cache'],weights_only=True);indices=data['unique_indices'][data['documents'][data['unique_indices']]<32];assert torch.equal(indices,s['indices']) and len(indices)==4092
 stats={};errors={}
 for name,value in [('input',s['z'][indices].double()),('output',data['Y'][1][indices].double())]:
  mean=value.mean(0);scale=(value-mean).square().mean().sqrt();assert float(scale)>0
  array=value.numpy();nm=array.mean(0);ns=np.sqrt(np.square(array-nm).mean());assert np.allclose(mean.numpy(),nm,rtol=1e-12,atol=1e-12) and np.isclose(float(scale),ns,rtol=1e-12)
  stats[name+'_mean']=mean.float();stats[name+'_scale']=scale.float();errors[name]=dict(mean_max_difference=float(np.max(np.abs(mean.numpy()-nm))),scale_difference=abs(float(scale)-float(ns)))
 entries=[]
 for seed in [71,72]:
  for kind in KINDS:
   model=BilinearInputFeaturizer(kind,seed);path=A/f'emlp_m81_input_initial_{kind}_{seed}.pt';assert not path.exists();torch.save(dict(state=model.state_dict(),statistics=stats,kind=kind,seed=seed,protocol_sha256=sha(protocol),module_sha256=sha(P/'bilinear_input_featurizer.py'),training_indices=indices,optimizer_updates=0),path);entries.append(dict(kind=kind,seed=seed,path=str(path),sha256=sha(path)))
 result=dict(status='frozen_untrained',source_sha256=sha(__file__),protocol_sha256=sha(protocol),module_sha256=sha(P/'bilinear_input_featurizer.py'),architecture_checks_sha256=sha(V/'bilinear_input_architecture_checks.json'),training_positions=4092,training_documents=list(range(32)),statistics_dtype='float64 computation storedfloat32; fixed thereafter',statistics_audit=errors,models=entries,optimizer_updates=0)
 out=V/'bilinear_input_initial_states.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print('FROZEN',len(entries),'states')
if __name__=='__main__':main()

"""M95 independently score final fixed training examples; not transfer evaluation."""
import json
import numpy as np
import torch
from product_dictionary_featurizer import ProductDictionaryFeaturizer
from bilinear_input_causal_loss import InputCausalExperiment
from balanced_training_runtime import load_natural,natural_loss
from train_product_dictionary import intact_task
from fit_pca_encoder import sha,V,A

def lp(t):
 x=t.detach().cpu().double().numpy();x=x-x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);rp=V/'product_dictionary_training_results.json';r=json.loads(rp.read_text());audit=json.loads((V/'product_dictionary_training_audit.json').read_text());assert audit['results_sha256']==sha(rp);e=InputCausalExperiment();natural,_=load_natural(e);ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];ns=torch.load(ref['path'],weights_only=True);zn=ns['z'][:128].cuda()[None];key=json.loads((V/'training_causal_coverage_protocol.json').read_text())['pairs'][0]['id']+'|0';donor=e.donor_key(key);initials=json.loads((V/'product_dictionary_initial_states.json').read_text())['models'];rows=[];samples=[];maximum=0
 def check(a,b):
  nonlocal maximum
  assert np.isclose(a,b,rtol=1e-7,atol=1e-9),(a,b);maximum=max(maximum,abs(a-b))
 for entry in r['models']:
  assert sha(entry['path'])==entry['sha256'];s=torch.load(entry['path'],weights_only=True);f=ProductDictionaryFeaturizer(s['state']).cuda();qr=f(e.input_cache[key]['input'].cuda())[0];qd=f(e.input_cache[donor]['input'].cuda())[0];K,B,intact,branches=e.joint_losses(qr,qd,key);Kcontrol=intact_task(e,qr,key);assert torch.equal(K,Kcontrol);c=e.cache[key];n=lp(c['native']);prob=np.exp(n);check(float((prob*(n-lp(intact))).sum()),float(K));bb=[]
  for branch in branches:
   teacher=next(t for t in c['targets'] if t['region']==branch['region']);target=lp(teacher['native_changed'])-n;response=lp(branch['changed'])-lp(branch['base']);R=float((prob*(response-target)**2).sum());check(R,float(branch['R']))
   if branch['region']=='identity':assert R==0
   else:bb.append(R/e.den[c['family'],branch['region']])
  check(float(np.mean(bb)),float(B));qn=f(zn)[0];NK,nout=natural_loss(e,qn,natural[0]);nlp=lp(natural[0]['native']);check(float((np.exp(nlp)*(nlp-lp(nout))).sum(-1).mean()),float(NK));rows.append(dict(kind=entry['kind'],seed=entry['seed'],arm=entry['arm'],key=key,task_K=float(K),task_B=float(B),natural_document=0,natural_K=float(NK),intact_K_path_exact=True));samples.append(dict(row=len(rows)-1,intact=intact.cpu(),natural=nout.cpu(),branches=[dict(region=b['region'],corner=b['corner'],base=b['base'].cpu(),changed=b['changed'].cpu()) for b in branches]));print('FINAL_LOSS',entry['kind'],entry['seed'],entry['arm'],flush=True)
 path=A/'emlp_m95_final_loss_samples.pt';assert not path.exists();torch.save(samples,path);out=dict(status='passed',source_sha256=sha(__file__),results_sha256=sha(rp),structural_audit_sha256=sha(V/'product_dictionary_training_audit.json'),rows=rows,max_metric_absolute_error=maximum,forward_count=e.calls,samples_path=str(path),samples_sha256=sha(path),scope='All4finalmodels on firsttrainingpair endpoint0 and naturaltrainingdoc0. Independent NumPy taskK/threecornerR/balancedB/naturalK; intactonlycodepath exactly matchesjointK. Training-only fixed arithmetic check, not evaluated quality ortransfer.')
 dest=V/'product_dictionary_final_loss_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('DONE',e.calls,maximum)
if __name__=='__main__':main()

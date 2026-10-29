"""M99 independent grouped-gradient population recomputation and schedule audit."""
import json
from collections import Counter
import numpy as np
import torch
from fit_pca_encoder import V,A,sha
from bilinear_input_causal_loss import InputCausalExperiment
from balanced_training_runtime import load_natural,natural_loss
from product_dictionary_featurizer import ProductDictionaryFeaturizer

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');ap=V/'product_dictionary_component_audit_protocol.json';q=json.loads(ap.read_text());assert q['source_sha256']==sha(__file__);rp=V/'product_dictionary_component_gradients.json';r=json.loads(rp.read_text());pp=V/'product_dictionary_component_execution.json';p=json.loads(pp.read_text());assert r['receipt']['protocol_sha256']==sha(pp)
 for name,digest in p['source_hashes'].items():assert sha(V.parent/name)==digest
 for name,digest in p['references'].items():assert sha(V/name)==digest
 e=InputCausalExperiment();natural,_=load_natural(e);ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];zref=torch.load(ref['path'],weights_only=True)['z'];schedules=json.loads((V/'product_dictionary_coverage_schedules.json').read_text())['schedules'];rows=[];arrays=[]
 assert len(r['models'])==4 and {(x['seed'],x['point']) for x in r['models']}=={(x['seed'],x['point']) for x in p['models']}
 for ent in r['models']:
  assert sha(ent['path'])==ent['sha256'];data=torch.load(ent['path'],weights_only=True);entry=next(x for x in p['models'] if (x['seed'],x['point'])==(ent['seed'],ent['point']));assert data['entry']==entry and data['receipt']==r['receipt'] and data['completed']==140;assert sha(entry['path'])==entry['sha256'];f=ProductDictionaryFeaturizer(torch.load(entry['path'],weights_only=True)['state']).cuda();schedule=schedules[str(entry['seed'])]['full_pool'];keys=[x['pair_id']+f'|{ep}' for x in schedule['task_updates'] for ep in (0,1)];assert [x['key'] for x in data['rows']]==keys;counts=Counter(schedule['natural_documents']);assert data['natural_document_counts']==dict(counts);sums={n:np.zeros((128,4608),dtype=np.float64) for n in ['natural_K','task_K','source','final']};losses={n:0. for n in sums}
  def add(name,loss,weight):
   if loss is None:return
   g=torch.autograd.grad(loss,f.coefficients,retain_graph=True)[0];sums[name]+=weight*g.detach().cpu().numpy().astype(np.float64);losses[name]+=weight*float(loss.detach())
  for doc in sorted(counts):
   pred=f(zref[doc*128:(doc+1)*128].cuda()[None])[0];loss,_=natural_loss(e,pred,natural[doc]);add('natural_K',loss,counts[doc]/70);del pred,loss
  for i,key in enumerate(keys):
   qr=f(e.input_cache[key]['input'].cuda())[0];qd=f(e.input_cache[e.donor_key(key)]['input'].cuda())[0];K,B,_,branches=e.joint_losses(qr,qd,key);active=[x for x in branches if x['region']!='identity'];row=data['rows'][i];assert row['index']==i and row['family']==e.cache[key]['family'] and row['nonidentity_branches']==len(active);assert float(K.detach())==row['losses']['task_K'] and float(B.detach())==row['B'];add('task_K',K,1/140)
   for name,region in [('source','changed_input_positions'),('final','final_position')]:
    selected=[x['R']/e.den[e.cache[key]['family'],region] for x in active if x['region']==region]
    add(name,torch.stack(selected).sum()/len(active) if selected else None,1/140)
   del qr,qd,K,B,branches,active,selected
  checks=[]
  for name in sums:
   wanted=data['gradients'][name].numpy() if name.endswith('_K') else sum(t.numpy() for n,t in data['gradients'].items() if n.startswith(name+'__'));wantedloss=data['losses'][name] if name.endswith('_K') else sum(x for n,x in data['losses'].items() if n.startswith(name+'__'));error=float(np.linalg.norm(sums[name]-wanted));norm=float(np.linalg.norm(wanted));allowed=q['gradient_atol']+q['gradient_rtol']*norm;assert error<=allowed,(ent,name,error,allowed);assert abs(losses[name]-wantedloss)<1e-12+1e-12*abs(wantedloss);checks.append(dict(component=name,norm=norm,absolute_error=error,relative_error=error/max(norm,1e-300),allowed=allowed,loss=losses[name]))
  rows.append(dict(seed=ent['seed'],point=ent['point'],checks=checks));arrays.append(dict(seed=ent['seed'],point=ent['point'],gradients={n:torch.from_numpy(x) for n,x in sums.items()}));print('AUDITED',ent['seed'],ent['point'],flush=True)
 path=A/'emlp_m99_component_grouped_audit.pt';assert not path.exists();torch.save(arrays,path);out=dict(status='passed',source_sha256=sha(__file__),protocol_sha256=sha(ap),result_sha256=sha(rp),rows=rows,forward_count=e.calls,samples_path=str(path),samples_sha256=sha(path),scope='All4populationgradients independently accumulated with NumPy; source/final differentiated as groupedlosses rather than sumofindividualgradients. Exactscalar replay/schedules. Same nativeautograd, notindependentJacobian. Frozenfloat32VJPtolerance retained.');f=V/'product_dictionary_component_gradient_audit.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print('DONE',e.calls)
if __name__=='__main__':main()

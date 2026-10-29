"""M99 fixed-state training-population gradients, with atomic endpoint checkpoints."""
import json
from collections import Counter
import torch
from fit_pca_encoder import V,A,sha
from bilinear_input_causal_loss import InputCausalExperiment
from balanced_training_runtime import load_natural,natural_loss
from product_dictionary_featurizer import ProductDictionaryFeaturizer
from product_dictionary_component_gradients import components,gradient,NAMES

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=V/'product_dictionary_component_execution.json';p=json.loads(pp.read_text())
 for n,h in p['source_hashes'].items():assert sha(V.parent/n)==h
 for n,h in p['references'].items():assert sha(V/n)==h
 assert json.loads((V/'product_dictionary_component_preflight.json').read_text())['status']=='passed'
 e=InputCausalExperiment();natural,np=load_natural(e);ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];ns=torch.load(ref['path'],weights_only=True);schedules=json.loads((V/'product_dictionary_coverage_schedules.json').read_text())['schedules'];entries=[]
 receipt=dict(protocol_sha256=sha(pp),source_sha256=sha(__file__),native_cache_sha256=sha(e.cachepath),natural_cache_sha256=sha(np),checkpoint_sha256=e.receipt.weights_sha256)
 for entry in p['models']:
  assert sha(entry['path'])==entry['sha256'];state=torch.load(entry['path'],weights_only=True)['state'];f=ProductDictionaryFeaturizer(state).cuda();schedule=schedules[str(entry['seed'])]['full_pool'];ident=f"{entry['seed']}_{entry['point']}";outpath=A/f'emlp_m99_components_{ident}.pt';progress=outpath.with_suffix('.progress.pt')
  if outpath.exists():
   out=torch.load(outpath,weights_only=True);assert out['receipt']==receipt and out['entry']==entry;entries.append(dict(seed=entry['seed'],point=entry['point'],path=str(outpath),sha256=sha(outpath)));continue
  names=('natural_K','task_K')+NAMES;sums={n:torch.zeros_like(f.coefficients,device='cpu',dtype=torch.float64) for n in names};losses={n:0. for n in names};rows=[];completed=0
  if progress.exists():
   old=torch.load(progress,weights_only=True);assert old['receipt']==receipt and old['entry']==entry;sums,losses,rows,completed=[old[k] for k in ['gradients','losses','rows','completed']]
  else:
   for doc,multiplicity in sorted(Counter(schedule['natural_documents']).items()):
    z=ns['z'][doc*128:(doc+1)*128].cuda()[None];prediction=f(z)[0];K,_=natural_loss(e,prediction,natural[doc]);weight=multiplicity/70;sums['natural_K']+=weight*gradient(K,f);losses['natural_K']+=weight*float(K.detach());del K,prediction
  keys=[a['pair_id']+f'|{ep}' for a in schedule['task_updates'] for ep in (0,1)];assert len(keys)==len(set(keys))==140
  for i,key in enumerate(keys):
   if i<completed:continue
   K,B,parts,count=components(e,f,key);allparts=dict(task_K=K,**parts);values={n:float(x.detach()) if x is not None else 0. for n,x in allparts.items()}
   assert abs(sum(values[n] for n in NAMES)-float(B.detach()))<1e-12+1e-12*abs(float(B.detach()))
   for n,x in allparts.items():sums[n]+=gradient(x,f)/140;losses[n]+=values[n]/140
   rows.append(dict(index=i,key=key,family=e.cache[key]['family'],nonidentity_branches=count,losses=values,B=float(B.detach())));del K,B,parts,allparts
   assert all(x.grad is None for x in e.model.parameters())
   if (i+1)%14==0:
    tmp=progress.with_suffix('.tmp');torch.save(dict(receipt=receipt,entry=entry,gradients=sums,losses=losses,rows=rows,completed=i+1),tmp);tmp.replace(progress);print('POINT',ident,'ENDPOINTS',i+1,flush=True)
  assert all(torch.equal(x.detach().cpu(),state[n]) for n,x in f.state_dict().items());assert all(torch.isfinite(x).all() for x in sums.values())
  tmp=outpath.with_suffix('.tmp');torch.save(dict(receipt=receipt,entry=entry,gradients=sums,losses=losses,rows=rows,completed=140,natural_document_counts=dict(Counter(schedule['natural_documents']))),tmp);tmp.replace(outpath);entries.append(dict(seed=entry['seed'],point=entry['point'],path=str(outpath),sha256=sha(outpath)));print('COMPLETE',ident,flush=True)
 dest=V/'product_dictionary_component_gradients.json';assert not dest.exists();dest.write_text(json.dumps(dict(status='completed_audit_pending',receipt=receipt,models=entries,forwards_this_process=e.calls,scope='Training-only fixed-state populationgradients. Natural repeateddocuments evaluated once with exactmultiplicity. No fit or heldoutgradients; Euclidean coefficientmetric, notAdam.'),indent=2)+'\n');print('DONE',len(entries),e.calls,flush=True)
if __name__=='__main__':main()

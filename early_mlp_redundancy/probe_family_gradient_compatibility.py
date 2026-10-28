"""M24 fixed starting-point gradient compatibility; no fitting or target selection."""
from pathlib import Path
import json,hashlib,collections
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from collect_response_targets import response_error
P=Path(__file__).resolve().parent/'validation'; A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 pp=P/'family_gradient_protocol.json';protocol=json.loads(pp.read_text());tp=P/'response_training_targets.json';t=json.loads(tp.read_text());assert sha(tp)==protocol['targets_sha256'];p=json.loads((P/'response_supervision_protocol.json').read_text());cp=Path(t['cache_path']);assert sha(cp)==t['cache_sha256'];cache=torch.load(cp,weights_only=True)
 op=P/'family_gradient_compatibility.json';assert not op.exists();model,receipt=load_model();model.requires_grad_(False);device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;calls=0;results=[];saved=[]
 cohort=json.loads((P/'natural_token_context_cohort.json').read_text());natural={}
 # Only documents actually used by the frozen training schedules.
 docs=sorted({s['natural_documents'][step] for s in p['schedules'].values() for step in range(0,p['steps'],2)})
 assert all(0<=d<32 for d in docs)
 for d in docs:
  ids=torch.tensor([cohort['token_ids'][d]],device=device);out,cap=trace(model,ids);calls+=1
  with torch.no_grad():write=mlp.Down(cap[1]['products'])
  natural[d]=dict(ids=ids.cpu(),native=out.cpu(),write=write.cpu())
 for entry in p['initial_models']:
  assert sha(Path(entry['path']))==entry['sha256'];s=torch.load(entry['path'],weights_only=True);f=Featurizer(entry['kind'],entry['seed']).to(device);f.load_state_dict(s['state']);mu=s['mean'].to(device);scale=s['scale'].to(device);params=list(f.parameters());weight=next(x['response_weight'] for x in t['calibration'] if (x['kind'],x['seed'])==(entry['kind'],entry['seed']));indices=[i for i,r in enumerate(t['records']) if (r['kind'],r['seed'])==(entry['kind'],entry['seed'])];counts=collections.Counter(t['records'][i]['family'] for i in indices);assert len(indices)==120 and len(counts)==10
  vectors={};losses=collections.defaultdict(float);checks=[]
  def flat(loss,retain=False):
   gs=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
   return torch.cat([(torch.zeros_like(v) if g is None else g).detach().flatten().double().cpu() for v,g in zip(params,gs)])
  def forward(c,target=None):
   nonlocal calls
   ids=c['ids'].to(device);rec,_,_=f((c['write'].to(device)-mu)/scale);q=rec*scale+mu
   out,_=trace.__wrapped__(model,ids,replace=lambda j,z,w:q+mlp.Down_bias if j==1 else w);calls+=1;teacher=c['native'].to(device);lp=teacher.double().log_softmax(-1);K=(lp.exp()*(lp-out.double().log_softmax(-1))).sum(-1).mean()
   if target is None:return K,None
   delta=target['delta'].to(device).detach()
   if bool((delta==0).all()):changed=out[0,-1]
   else:
    alt,_=trace.__wrapped__(model,ids,replace=lambda j,z,w:q-delta+mlp.Down_bias if j==1 else w);calls+=1;changed=alt[0,-1]
   R=response_error(teacher[0,-1],target['native_changed'].to(device).detach(),out[0,-1],changed)
   return K,R
  for ii,i in enumerate(indices):
   row=t['records'][i];c=cache['inputs'][row['input_key']];K,R=forward(c,cache['targets'][i]);gk=flat(K,True);gr=flat(R)
   assert abs(float(K.detach())-row['initial_K'])<1e-7 and abs(float(R.detach())-row['initial_R'])<1e-7
   for suffix,g,l in [('K',gk,K),('R',gr,R)]:
    key=row['family']+':'+suffix
    if key not in vectors:vectors[key]=torch.zeros_like(g)
    vectors[key]+=g/counts[row['family']];losses[key]+=float(l.detach())/counts[row['family']]
   if ii==0:
    # Parameter-space central differences for the combined objective; restore exactly.
    direction=gk+weight*gr;direction/=direction.norm();original=[v.detach().clone() for v in params];offset=0;ds=[]
    for v in params:ds.append(direction[offset:offset+v.numel()].reshape(v.shape).to(v));offset+=v.numel()
    analytic=float(torch.dot(gk+weight*gr,direction));fd=[]
    for eps in protocol['finite_difference_epsilons']:
     vals=[]
     for sign in [1,-1]:
      with torch.no_grad():
       for v,o,d in zip(params,original,ds):v.copy_(o+sign*eps*d)
       kk,rr=forward(c,cache['targets'][i]);vals.append(float(kk+weight*rr))
     numeric=(vals[0]-vals[1])/(2*eps);fd.append(dict(epsilon=eps,numeric=numeric,analytic=analytic,relative_error=abs(numeric-analytic)/max(abs(analytic),1e-30)))
    with torch.no_grad():
     for v,o in zip(params,original):v.copy_(o)
    assert min(x['relative_error'] for x in fd)<.03,fd
    checks.append(dict(target_index=i,finite_differences=fd))
   if ii%30==29:print('TARGETS',entry['kind'],entry['seed'],ii+1,flush=True)
  schedule=p['schedules'][str(entry['seed'])];dc=collections.Counter(schedule['natural_documents'][step] for step in range(0,p['steps'],2));gn=torch.zeros_like(gk)
  for d,n in dc.items():
   K,_=forward(natural[d]);gn+=flat(K)*n/60;losses['natural:K']+=float(K.detach())*n/60
  vectors['natural:K']=gn
  for family in counts:vectors[family+':combined']=vectors[family+':K']+weight*vectors[family+':R']
  vectors['task:K']=sum(vectors[a+':K']*n/120 for a,n in counts.items());vectors['task:weighted_R']=sum(vectors[a+':R']*weight*n/120 for a,n in counts.items());vectors['schedule:combined']=(gn+vectors['task:K']+vectors['task:weighted_R'])/2
  names=list(vectors);G=torch.stack([vectors[n] for n in names]);gram=G@G.T;norms=gram.diag().sqrt();denom=norms[:,None]*norms[None,:];cos=torch.where(denom>0,gram/denom,torch.zeros_like(gram));assert torch.isfinite(G).all();assert all(v.grad is None for v in model.parameters())
  results.append(dict(kind=entry['kind'],seed=entry['seed'],weight=weight,family_endpoint_counts=dict(counts),names=names,losses=dict(losses),norms=norms.tolist(),gram=gram.tolist(),cosine=cos.tolist(),finite_difference_checks=checks));saved.append(dict(kind=entry['kind'],seed=entry['seed'],names=names,gradients=G));print('MODEL_DONE',entry['kind'],entry['seed'],flush=True)
 sp=A/'emlp_m24_family_gradients.pt';torch.save(saved,sp);out=dict(protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),targets_sha256=sha(tp),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,models=results,samples_path=str(sp),samples_sha256=sha(sp),scope=protocol['scope']);op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls,flush=True)
if __name__=='__main__':main()

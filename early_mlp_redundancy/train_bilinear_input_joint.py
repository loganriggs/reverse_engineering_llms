"""M87 sixteen fixed matched input-featurizer supervision fits."""
import json
import torch
from torch.nn import functional as F
from bilinear_input_runtime import InputWritePredictor
from bilinear_input_causal_loss import InputCausalExperiment
from balanced_training_runtime import load_natural,natural_loss
from inspect_model import trace
from fit_pca_encoder import sha,V,A
P=V.parent

def intact_task(e,q,key):
 c=e.cache[key];out,_=trace.__wrapped__(e.model,c['ids'].cuda(),replace=lambda j,z,w:q+e.mlp.Down_bias if j==1 else w);e.calls+=1;nlp=c['native'].cuda().double().log_softmax(-1);return (nlp.exp()*(nlp-out[0,-1].double().log_softmax(-1))).sum()

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=V/'bilinear_input_joint_training_protocol.json';q=json.loads(pp.read_text())
 for name,digest in q['references'].items():assert sha(V/name)==digest
 assert json.loads((V/'bilinear_input_joint_calibration_audit.json').read_text())['status']=='passed';e=InputCausalExperiment();natural,np=load_natural(e);ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];ns=torch.load(ref['path'],weights_only=True);initials=json.loads((V/'bilinear_input_initial_states.json').read_text())['models']
 receipt=dict(protocol_sha256=sha(pp),source_sha256=sha(__file__),module_sha256=sha(P/'bilinear_input_featurizer.py'),runtime_sha256=sha(P/'bilinear_input_runtime.py'),loss_source_sha256=sha(P/'bilinear_input_causal_loss.py'),natural_runtime_sha256=sha(P/'balanced_training_runtime.py'),native_input_reference_sha256=sha(V/'native_input_reference.json'),task_input_sha256=sha(V/'input_training_inventory.json'),natural_cache_sha256=sha(np),native_cache_sha256=sha(e.cachepath),checkpoint_sha256=e.receipt.weights_sha256,torch_version=str(torch.__version__),cuda_version=torch.version.cuda)
 rp=V/'bilinear_input_joint_training_receipt.json'
 if rp.exists():assert json.loads(rp.read_text())==receipt
 else:rp.write_text(json.dumps(receipt,indent=2)+'\n')
 entries=[];calls_by_model=[]
 for initial in q['models']:
  assert sha(initial['path'])==initial['sha256'];warm=torch.load(initial['path'],weights_only=True);base=next(x for x in initials if (x['kind'],x['seed'])==(initial['kind'],initial['seed']));assert sha(base['path'])==base['sha256'];constructor=torch.load(base['path'],weights_only=True);schedule=e.protocol['schedules'][str(initial['seed'])]
  for arm in q['arms']:
   f=InputWritePredictor(constructor).cuda();f.load_state_dict(warm['state']);opt=torch.optim.Adam(f.parameters(),lr=q['learning_rate']);path=A/f"emlp_m87_{initial['kind']}_{initial['seed']}_{arm}.pt";progress=path.with_suffix('.progress.pt');history=[];start=0;calls_before=e.calls
   if path.exists():
    saved=torch.load(path,weights_only=True);assert saved['receipt']==receipt and saved['initial_sha256']==initial['sha256'] and len(saved['history'])==132;entries.append(dict(kind=initial['kind'],seed=initial['seed'],arm=arm,path=str(path),sha256=sha(path)));continue
   if progress.exists():
    saved=torch.load(progress,weights_only=True);assert saved['receipt']==receipt and saved['initial_sha256']==initial['sha256'];f.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer']);history=saved['history'];start=len(history)
   for step in range(start,132):
    opt.zero_grad();parts=[]
    if step%2==0:
     doc=schedule['natural_documents'][step//2];z=ns['z'][doc*128:(doc+1)*128].cuda()[None];prediction,support,_=f(z);K,_=natural_loss(e,prediction,natural[doc]);K.backward();parts.append(dict(key=f'natural:{doc}',K=float(K.detach()),B=None,loss=float(K.detach()),active_mean=float(support.sum(-1).mean())));del prediction,K
    else:
     assignment=schedule['task_updates'][step//2]
     for ep in [0,1]:
      key=assignment['pair_id']+f'|{ep}';assert e.cache[key]['family']==assignment['family'];qr,support,_=f(e.input_cache[key]['input'].cuda())
      if arm=='intact_KL':K=intact_task(e,qr,key);B=None;loss=K/2
      else:
       donor=e.donor_key(key);qd,_,_=f(e.input_cache[donor]['input'].cuda());result=e.joint_losses(qr,qd,key);K,B=result[:2];loss=(K+q['joint_weight']*B)/2
      assert torch.isfinite(loss);loss.backward();parts.append(dict(key=key,family=assignment['family'],K=float(K.detach()),B=float(B.detach()) if B is not None else None,loss=float(loss.detach()),active_mean=float(support.sum(-1).mean())))
      if arm!='intact_KL':del qd,result
      del qr,K,B,loss
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters());assert all(p.grad is None for p in e.model.parameters());opt.step()
    with torch.no_grad():f.features.dictionary.copy_(F.normalize(f.features.dictionary,dim=0))
    history.append(dict(step=step,parts=parts,loss=sum(p['loss'] for p in parts)))
    if (step+1)%11==0:
     temp=progress.with_suffix('.tmp');torch.save(dict(receipt=receipt,initial_sha256=initial['sha256'],state=f.state_dict(),optimizer=opt.state_dict(),history=history),temp);temp.replace(progress)
    if (step+1)%66==0:print(initial['kind'],initial['seed'],arm,step+1,flush=True)
   state=dict(receipt=receipt,initial_sha256=initial['sha256'],kind=initial['kind'],seed=initial['seed'],arm=arm,state={k:v.detach().cpu() for k,v in f.state_dict().items()},history=history,optimizer_updates=132,optimizer_defaults={k:v for k,v in opt.param_groups[0].items() if k!='params'});temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path);entries.append(dict(kind=initial['kind'],seed=initial['seed'],arm=arm,path=str(path),sha256=sha(path)));calls_by_model.append(dict(kind=initial['kind'],seed=initial['seed'],arm=arm,calls_this_process=e.calls-calls_before));print('FIT_COMPLETE',initial['kind'],initial['seed'],arm,flush=True)
 out=V/'bilinear_input_joint_training_results.json';assert not out.exists();out.write_text(json.dumps(dict(status='fits_completed_audit_pending',receipt=receipt,models=entries,optimizer_updates=2112,native_forwards_this_process=e.calls,calls_by_model=calls_by_model,scope='All16fixedfinalstates retained. Online losses are not evaluated fidelity ortransfer. No modelselection.'),indent=2)+'\n');print('DONE',len(entries),e.calls,flush=True)
if __name__=='__main__':main()

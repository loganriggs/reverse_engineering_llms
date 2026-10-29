"""M112 matched hard/continuous block fits; identical causal supervision and schedule."""
import json
import torch
from torch.nn import functional as F
from encoded_change_loss import EncodedExperiment
from balanced_response_loss import P,V,A,sha
from balanced_training_runtime import load_natural,natural_loss
from continuous_write_featurizer import ContinuousBlockFeaturizer
from fit_pca_encoder import sha


def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
 pp=V/'continuous_training_protocol.json';p=json.loads(pp.read_text());assert p['runner_sha256']==sha(__file__)
 for name,h in p['dependencies'].items():assert sha(P/name)==h
 for name,h in p['bindings'].items():assert sha(V/name)==h
 assert json.loads((V/'continuous_causal_gradient_checks.json').read_text())['status']=='passed'
 e=EncodedExperiment();natural,np=load_natural(e);cal=json.loads((V/'encoded_change_calibration.json').read_text());assert cal['encoded_weight']==p['causal_weight']
 previous=json.loads((V/'encoded_change_training_results.json').read_text())['models']
 receipt=dict(protocol_sha256=sha(pp),evaluation_protocol_sha256=sha(V/'continuous_training_evaluation_protocol.json'),runner_sha256=sha(__file__),checkpoint_sha256=e.receipt.weights_sha256,task_cache_sha256=sha(e.cachepath),natural_cache_sha256=sha(np),torch_version=str(torch.__version__),cuda_version=torch.version.cuda)
 dest=A/'emlp_m112_continuous_training';dest.mkdir(exist_ok=True);op=V/'continuous_training_results.json';assert not op.exists();models=[]
 for initial in p['starts']:
  seed=initial['seed'];schedule=e.protocol['schedules'][str(seed)]
  for mode in p['modes']:
   path=dest/f'block_{seed}_{mode}.pt';progress=path.with_suffix('.progress.pt')
   if path.exists():
    saved=torch.load(path,weights_only=True);assert saved['receipt']==receipt and len(saved['history'])==132
    models.append(dict(seed=seed,mode=mode,path=str(path),sha256=sha(path),hard_replay=saved['hard_replay']));continue
   f,mu,scale=e.featurizer(initial)
   if mode=='continuous':
    new=ContinuousBlockFeaturizer(seed).cuda();new.load_state_dict(f.state_dict());f=new
   opt=torch.optim.Adam(f.parameters(),lr=e.protocol['learning_rate']);history=[]
   if progress.exists():
    state=torch.load(progress,weights_only=True);assert state['receipt']==receipt and state['initial_sha256']==initial['sha256'];f.load_state_dict(state['state']);opt.load_state_dict(state['optimizer']);history=state['history']
   for step in range(len(history),132):
    opt.zero_grad();parts=[]
    if step%2==0:
     doc=schedule['natural_documents'][step//2];c=natural[doc];pred,support,code=f((c['write'].cuda()-mu)/scale);K,_=natural_loss(e,pred*scale+mu,c);K.backward();parts.append(dict(key=f'natural:{doc}',K=float(K.detach()),loss=float(K.detach()),active_mean=float(support.sum(-1).mean()),code_energy=float(code.detach().square().mean())))
    else:
     assignment=schedule['task_updates'][step//2]
     for ep in [0,1]:
      key=assignment['pair_id']+f'|{ep}';c=e.cache[key];d=e.cache[e.donor_key(key)];assert c['family']==assignment['family'];pr,support,code=f((c['write'].cuda()-mu)/scale);pd,ds,_=f((d['write'].cuda()-mu)/scale);K,R,B,_,_=e.encoded_losses(pr*scale+mu,pd*scale+mu,key);loss=(K+p['causal_weight']*B)/2;assert torch.isfinite(loss);loss.backward();parts.append(dict(key=key,donor_key=e.donor_key(key),family=c['family'],K=float(K.detach()),raw_R=float(R.detach()),balanced_R=float(B.detach()),weight=p['causal_weight'],loss=float(loss.detach()),active_mean=float(support.sum(-1).mean()),donor_active_mean=float(ds.sum(-1).mean()),code_energy=float(code.detach().square().mean())))
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in f.parameters());assert all(x.grad is None for x in e.model.parameters());opt.step()
    with torch.no_grad():f.dictionary.copy_(F.normalize(f.dictionary,dim=0))
    history.append(dict(step=step,parts=parts,loss=sum(x['loss'] for x in parts)))
    if (step+1)%11==0:
     tmp=progress.with_suffix('.tmp');torch.save(dict(receipt=receipt,initial_sha256=initial['sha256'],state=f.state_dict(),optimizer=opt.state_dict(),history=history),tmp);tmp.replace(progress);print('UPDATES',seed,mode,step+1,flush=True)
   replay=None
   if mode=='hard':
    control=next(x for x in previous if x['kind']=='block' and x['seed']==seed);assert sha(control['path'])==control['sha256'];old=torch.load(control['path'],weights_only=True)
    diffs={k:float((f.state_dict()[k].cpu()-old['state'][k]).abs().max()) for k in old['state']};replay=dict(reference_sha256=control['sha256'],max_abs_parameter_difference=max(diffs.values()),bit_exact=all(torch.equal(f.state_dict()[k].cpu(),old['state'][k]) for k in old['state']))
   tmp=path.with_suffix('.tmp');torch.save(dict(receipt=receipt,initial_sha256=initial['sha256'],kind='block',seed=seed,mode=mode,state={k:v.detach().cpu() for k,v in f.state_dict().items()},mean=mu.cpu(),scale=scale.cpu(),history=history,hard_replay=replay),tmp);tmp.replace(path);models.append(dict(seed=seed,mode=mode,path=str(path),sha256=sha(path),hard_replay=replay));print('FIT_COMPLETE',seed,mode,'REPLAY',replay,flush=True)
 out=dict(status='fits_complete_evaluation_pending',receipt=receipt,models=models,optimizer_updates=528,forwards_this_process=e.calls,scope='All four final states retained. Matched 132-step adaptation from common historical hard-trained starts; not random initialization comparison. No held-out outcomes yet.')
 op.write_text(json.dumps(out,indent=2)+'\n');print('COMPLETE',len(models),flush=True)
if __name__=='__main__':main()

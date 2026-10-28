"""M27 full-data K-only controls at fixed M25 displacement norms."""
from pathlib import Path
import hashlib,json
import numpy as np,torch
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(6);pp=P/'aggregate_kl_protocol.json';p=json.loads(pp.read_text());gp=P/'family_gradient_compatibility.json';gr=json.loads(gp.read_text());assert sha(gp)==p['M24_sha256'];assert json.loads((P/'family_gradient_audit.json').read_text())['result_sha256']==sha(gp);sp=Path(gr['samples_path']);assert sha(sp)==gr['samples_sha256'];gradients=torch.load(sp,weights_only=True);rp=P/'finite_gradient_steps.json';r=json.loads(rp.read_text());assert sha(rp)==p['M25_sha256'];sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];saved=torch.load(sp,weights_only=True);old=json.loads((P/'finite_candidate_models.json').read_text());assert json.loads((P/'finite_candidate_models_audit.json').read_text())['result_sha256']==sha(P/'finite_candidate_models.json');dest=A/'emlp_m27_aggregate_kl_controls';assert not dest.exists();dest.mkdir();models=[];checks=[]
 with torch.no_grad():
  for e in p['initial_models']:
   assert sha(Path(e['path']))==e['sha256'];s=torch.load(e['path'],weights_only=True);g=next(x for x in gradients if (x['kind'],x['seed'])==(e['kind'],e['seed']));lookup={n:g['gradients'][i] for i,n in enumerate(g['names'])};direction=(lookup['natural:K']+lookup['task:K'])/2;ds=next(x for x in saved if 'parameter_deltas' in x and (x['kind'],x['seed'])==(e['kind'],e['seed']));norm=ds['raw_adam_delta'].norm();baseline=next(x for x in old['models'] if (x['kind'],x['seed'],x['arm'])==(e['kind'],e['seed'],'initial'));assert sha(Path(baseline['path']))==baseline['sha256'];models.append(baseline)
   for arm in p['arms']:
    mult=float(arm.split(':')[1]);raw=-direction/direction.norm()*norm*mult;f=Featurizer(e['kind'],e['seed']).cuda();f.load_state_dict(s['state']);original=[v.detach().clone() for v in f.parameters()];off=0;np_error=0
    for v,o in zip(f.parameters(),original):
     d=raw[off:off+v.numel()].reshape(v.shape).to(v);v.add_(d);off+=v.numel()
    f.dictionary.copy_(torch.nn.functional.normalize(f.dictionary,dim=0));off=0
    # Independent NumPy construction of every parameter tensor (float32 operations).
    for name,v in f.named_parameters():
     o=s['state'][name].numpy();d=raw[off:off+v.numel()].numpy().reshape(o.shape).astype(np.float32);expected=o+d;off+=v.numel()
     if name=='dictionary':expected=expected/np.sqrt(np.sum(expected**2,axis=0,keepdims=True))
     err=float(np.max(np.abs(expected-v.cpu().numpy())));np_error=max(np_error,err);assert err<1e-7,(name,err)
    actual=torch.cat([(v-o).flatten().double().cpu() for v,o in zip(f.parameters(),original)]);assert abs(float(raw.norm()/(norm*mult))-1)<1e-12;path=dest/f"{e['kind']}_{e['seed']}_{arm.replace(':','_')}.pt";torch.save(dict(state={k:v.cpu() for k,v in f.state_dict().items()},mean=s['mean'],scale=s['scale'],kind=e['kind'],seed=e['seed'],arm=arm),path);models.append(dict(kind=e['kind'],seed=e['seed'],arm=arm,path=str(path),sha256=sha(path)));checks.append(dict(kind=e['kind'],seed=e['seed'],arm=arm,raw_norm=float(raw.norm()),post_normalization_norm=float(actual.norm()),parameter_numpy_max_error=np_error,predicted_K_change=float(torch.dot(direction,actual)),predicted_weighted_R_change=float(torch.dot(lookup['task:weighted_R'],actual))));print('BUILT',e['kind'],e['seed'],arm,flush=True)
 out=dict(protocol_sha256=sha(pp),models=models,M24_sha256=sha(gp),M25_sha256=sha(rp));op=P/'aggregate_kl_models.json';op.write_text(json.dumps(out,indent=2)+'\n');(P/'aggregate_kl_models_audit.json').write_text(json.dumps(dict(status='passed',result_sha256=sha(op),checks=checks,scope='Every parameter independently reconstructed with NumPy, same raw norm as M25, immutable gradient/source hashes; initial candidates reused from exact-replayed M26. New controls are deliberately new states, not expected to match old training logits.'),indent=2)+'\n')
if __name__=='__main__':main()

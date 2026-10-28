"""M26 reconstruct all prespecified live M25 candidates; verify saved output replay."""
from pathlib import Path
import hashlib,json
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');pp=P/'finite_candidate_protocol.json';p=json.loads(pp.read_text());rp=P/'finite_gradient_steps.json';r=json.loads(rp.read_text());assert sha(rp)==p['M25_sha256'];assert json.loads((P/'finite_gradient_audit.json').read_text())['result_sha256']==sha(rp);sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);t=json.loads((P/'response_training_targets.json').read_text());cp=Path(t['cache_path']);assert sha(cp)==t['cache_sha256'];cache=torch.load(cp,weights_only=True);dest=A/'emlp_m26_finite_candidates';assert not dest.exists();dest.mkdir();models=[];checks=[];model,receipt=load_model();device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;calls=0
 with torch.no_grad():
  for e in p['initial_models']:
   assert sha(Path(e['path']))==e['sha256'];s=torch.load(e['path'],weights_only=True);ds=next(x for x in samples if 'parameter_deltas' in x and (x['kind'],x['seed'])==(e['kind'],e['seed']));mu=s['mean'].to(device);scale=s['scale'].to(device)
   for arm in ['initial']+p['arms']:
    label='initial:0.0:live' if arm=='initial' else arm;delta=next(x['delta'] for x in ds['parameter_deltas'] if x['label']==label);f=Featurizer(e['kind'],e['seed']).to(device);f.load_state_dict(s['state']);off=0
    original=[v.detach().clone() for v in f.parameters()];raw=ds['raw_adam_delta'];agg=ds['aggregate_gradient'];proposed=torch.zeros_like(raw) if arm=='initial' else (raw if arm.startswith('minibatch_adam:') else -agg/agg.norm()*raw.norm())*float(arm.split(':')[1])
    for v in f.parameters():v.add_(proposed[off:off+v.numel()].reshape(v.shape).to(v));off+=v.numel()
    if arm!='initial':f.dictionary.copy_(torch.nn.functional.normalize(f.dictionary,dim=0))
    actual=torch.cat([(v-o).flatten().double().cpu() for v,o in zip(f.parameters(),original)]);assert torch.equal(actual,delta)
    assert off==delta.numel();path=dest/f"{e['kind']}_{e['seed']}_{arm.replace(':','_')}.pt";torch.save(dict(state={k:v.cpu() for k,v in f.state_dict().items()},mean=s['mean'],scale=s['scale'],kind=e['kind'],seed=e['seed'],arm=arm),path);models.append(dict(kind=e['kind'],seed=e['seed'],arm=arm,path=str(path),sha256=sha(path)))
    sample=next(x for x in samples if 'row' in x and all(r['records'][x['row']][k]==val for k,val in [('kind',e['kind']),('seed',e['seed']),('condition',label)]));row=r['records'][sample['row']];i=row['target_index'];c=cache['inputs'][t['records'][i]['input_key']];ids=c['ids'].to(device);q=f((c['write'].to(device)-mu)/scale)[0]*scale+mu;out,_=trace(model,ids,replace=lambda j,z,w:q+mlp.Down_bias if j==1 else w);calls+=1;err=float((out.cpu()-sample['output']).abs().max());assert err==0,err
    delta_write=cache['targets'][i]['delta'].to(device);out,_=trace(model,ids,replace=lambda j,z,w:q-delta_write+mlp.Down_bias if j==1 else w);calls+=1;err2=float((out[0,-1].cpu()-sample['changed']).abs().max());assert err2==0,err2;checks.append(dict(kind=e['kind'],seed=e['seed'],arm=arm,output_max_error=err,response_output_max_error=err2));print('REPLAY',e['kind'],e['seed'],arm,flush=True)
 out=dict(protocol_sha256=sha(pp),models=models,M25_sha256=sha(rp),checkpoint_sha256=receipt.weights_sha256);op=P/'finite_candidate_models.json';op.write_text(json.dumps(out,indent=2)+'\n');(P/'finite_candidate_models_audit.json').write_text(json.dumps(dict(status='passed',result_sha256=sha(op),checks=checks,forward_count=calls,scope='All20 candidate checkpoint hashes and exact replay of intact/intervened output tensors on one fixed training endpoint per candidate. Not full training replay or independent transformer implementation.'),indent=2)+'\n')
if __name__=='__main__':main()

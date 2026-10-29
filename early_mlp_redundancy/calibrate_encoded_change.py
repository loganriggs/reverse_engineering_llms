"""M68 training-only initial loss calibration; no optimizer updates."""
from pathlib import Path
import json,torch,numpy as np
from encoded_change_loss import EncodedExperiment
from balanced_response_loss import P,V,A,sha

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);pp=V/'encoded_change_objective_preflight_protocol.json';q=json.loads(pp.read_text())
 for n,k in [('balanced_response_training_protocol.json','reference_training_protocol_sha256'),('balanced_response_calibration.json','reference_calibration_sha256'),('balanced_response_training_results.json','reference_final_models_sha256'),('training_causal_coverage.json','native_training_sha256')]:assert sha(V/n)==q[k]
 e=EncodedExperiment();oldchecks=json.loads((V/'balanced_response_gradient_checks.json').read_text());assert sha(e.cachepath)==oldchecks['cache_sha256'] and sha(P/'balanced_response_loss.py')==oldchecks['loss_source_sha256'];op=V/'encoded_change_calibration.json';assert not op.exists();dest=A/'emlp_m68_calibration';dest.mkdir(exist_ok=True);rows=[];files=[];identity=[];addresses=0
 # Validate every eligible source/final address using native writes, including unequal lengths.
 for key,c in e.cache.items():
  d=e.cache[e.donor_key(key)]
  for t in c['targets']:
   pos,dp=e.addresses[key,t['region']];delta=torch.zeros_like(c['write']);delta[:,pos]=d['write'][:,dp]-c['write'][:,pos];assert torch.equal(delta,t['delta']);addresses+=1
 for entry in e.protocol['models']:
  f,mu,scale=e.featurizer(entry);key0=e.protocol['schedules'][str(entry['seed'])]['task_updates'][0]['pair_id']+'|0';c=e.cache[key0];d=e.cache[e.donor_key(key0)];ik,ir,ib,iz,branches=e.encoded_losses(c['write'].cuda(),d['write'].cuda(),key0);assert float(ik)==float(ir)==float(ib)==0.;assert torch.equal(iz.cpu(),c['native']);assert all(torch.equal(b['changed'].cpu(),t['native_changed']) for b,t in zip(branches,c['targets']));identity.append(dict(kind=entry['kind'],seed=entry['seed'],key=key0,K=float(ik),raw=float(ir),balanced=float(ib)))
  # Deduplicate identical scheduled recipients only for forward work; retain every scheduled row.
  computed={};blob={}
  for ti,assignment in enumerate(e.protocol['schedules'][str(entry['seed'])]['task_updates']):
   for ep in [0,1]:
    key=assignment['pair_id']+f'|{ep}';c=e.cache[key];d=e.cache[e.donor_key(key)]
    if key not in computed:
     qr=f((c['write'].cuda()-mu)/scale)[0]*scale+mu;qd=f((d['write'].cuda()-mu)/scale)[0]*scale+mu;K,R,B,z,branches=e.encoded_losses(qr,qd,key);computed[key]=dict(K=float(K),R_raw=float(R),R_balanced=float(B));blob[key]=dict(intact=z.cpu(),qr=qr.cpu(),qd=qd.cpu(),branches=[dict(region=b['region'],delta=b['delta'].cpu(),changed=b['changed'].cpu()) for b in branches])
    vals=computed[key];old=next(x for x in e.cal['rows'] if x['kind']==entry['kind'] and x['seed']==entry['seed'] and x['task_update']==ti and x['key']==key);assert vals['K']==old['K'];rows.append(dict(kind=entry['kind'],seed=entry['seed'],task_update=ti,key=key,family=c['family'],**vals))
  path=dest/(entry['kind']+'_'+str(entry['seed'])+'.pt');torch.save(blob,path);files.append(dict(kind=entry['kind'],seed=entry['seed'],path=str(path),sha256=sha(path)));print('CALIBRATED',entry['kind'],entry['seed'],'FORWARDS',e.calls,flush=True)
 assert len(rows)==792 and addresses==418;K=sum(x['K'] for x in rows)/len(rows);B=sum(x['R_balanced'] for x in rows)/len(rows);weight=K/B
 op.write_text(json.dumps(dict(status='completed',source_sha256=sha(Path(__file__)),loss_source_sha256=sha(P/'encoded_change_loss.py'),protocol_sha256=sha(pp),reference_protocol_sha256=sha(e.pp),cache_sha256=sha(e.cachepath),rows=rows,files=files,identity_checks=identity,address_checks=addresses,mean_K=K,mean_R_balanced=B,encoded_weight=weight,forward_count=e.calls),indent=2)+'\n');print('DONE',e.calls,'WEIGHT',weight,flush=True)
if __name__=='__main__':main()

"""M86 fixed training-only global scale for all-three-corner supervision."""
import json
import torch
from bilinear_input_runtime import InputWritePredictor
from bilinear_input_causal_loss import InputCausalExperiment
from fit_pca_encoder import sha,V,A
P=V.parent

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);pp=V/'bilinear_input_joint_calibration_protocol.json';p=json.loads(pp.read_text())
 for name,digest in p['references'].items():assert sha(V/name)==digest
 e=InputCausalExperiment();initials=json.loads((V/'bilinear_input_initial_states.json').read_text())['models'];rows=[];files=[];dest=A/'emlp_m86_calibration';dest.mkdir(exist_ok=True)
 for entry in p['models']:
  assert sha(entry['path'])==entry['sha256'];warm=torch.load(entry['path'],weights_only=True);start=next(x for x in initials if (x['kind'],x['seed'])==(entry['kind'],entry['seed']));assert sha(start['path'])==start['sha256'];f=InputWritePredictor(torch.load(start['path'],weights_only=True)).cuda();f.load_state_dict(warm['state']);samples=[]
  for assignment_index,assignment in enumerate(e.protocol['schedules'][str(entry['seed'])]['task_updates']):
   for ep in [0,1]:
    key=assignment['pair_id']+f'|{ep}';donor=e.donor_key(key);assert e.cache[key]['family']==assignment['family'];qr,_,_=f(e.input_cache[key]['input'].cuda());qd,_,_=f(e.input_cache[donor]['input'].cuda());K,B,intact,branches=e.joint_losses(qr,qd,key);assert torch.isfinite(K) and torch.isfinite(B);row=dict(kind=entry['kind'],seed=entry['seed'],assignment=assignment_index,key=key,family=assignment['family'],K=float(K),B=float(B));rows.append(row);samples.append(dict(row=len(rows)-1,intact=intact.cpu(),branches=[dict(region=b['region'],corner=b['corner'],base=b['base'].cpu(),changed=b['changed'].cpu(),R=float(b['R'])) for b in branches]))
  path=dest/f"{entry['kind']}_{entry['seed']}.pt";assert not path.exists();torch.save(samples,path);files.append(dict(kind=entry['kind'],seed=entry['seed'],path=str(path),sha256=sha(path)));print('CALIBRATED',entry['kind'],entry['seed'],len(rows),flush=True)
 assert len(rows)==1056;K=sum(r['K'] for r in rows)/len(rows);B=sum(r['B'] for r in rows)/len(rows);assert K>0 and B>0
 result=dict(status='calibrated_audit_pending',source_sha256=sha(__file__),loss_source_sha256=sha(P/'bilinear_input_causal_loss.py'),runtime_sha256=sha(P/'bilinear_input_runtime.py'),protocol_sha256=sha(pp),rows=rows,files=files,mean_K=K,mean_B=B,joint_weight=K/B,forward_count=e.calls,optimizer_updates=0,scope='One globaltraining-only scale; repeatedscheduledrowsretained; allfullvectorssaved forindependentaudit. Noheldoutscoreorfit.')
 out=V/'bilinear_input_joint_calibration.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print('DONE',K,B,K/B,e.calls,flush=True)
if __name__=='__main__':main()

"""M112 fixed task fidelity and candidate-only norm-matched hard controls."""
import json,sys,hashlib
import torch
from fit_pca_encoder import V,A,sha
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from continuous_write_featurizer import ContinuousBlockFeaturizer


def margin(z,g,v):
 z=z.double();return float(torch.logsumexp(z[g[v]],0)-torch.logsumexp(z[g[1-v]],0))


def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 cohort=sys.argv[1];assert cohort in ['train','source','whole'];xp=V/'continuous_tasks_execution.json';execution=json.loads(xp.read_text());assert execution['source_sha256']==sha(__file__)
 for name,h in execution['bindings'].items():assert sha(V/name)==h
 for name,h in execution['helpers'].items():assert sha(V.parent/name)==h
 mr=json.loads((V/'continuous_training_results.json').read_text());assert json.loads((V/'continuous_training_audit.json').read_text())['results_sha256']==sha(V/'continuous_training_results.json')
 protocol=json.loads((V/'block_change_roles_protocol.json').read_text());stem={'train':'training_causal_coverage','source':'source_holdout_native','whole':'whole_task_native'}[cohort];nr=json.loads((V/(stem+'.json')).read_text());pairs=json.loads((V/(stem+'_protocol.json')).read_text())['pairs'];byid={x['id']:x for x in pairs};wanted=protocol[{'train':'training_pair_ids','source':'source_group_pair_ids','whole':'whole_task_pair_ids'}[cohort]];pairs=[byid[x] for x in wanted];assert sha(nr['samples_path'])==nr['samples_sha256'];native=torch.load(nr['samples_path'],weights_only=True)
 model,receipt=load_model();assert receipt.weights_sha256==nr['checkpoint_sha256'];mlp=model.transformer.h[1].mlp;models=[]
 for e in mr['models']:
  assert sha(e['path'])==e['sha256'];s=torch.load(e['path'],weights_only=True);f=(Featurizer('block',e['seed']) if e['mode']=='hard' else ContinuousBlockFeaturizer(e['seed'])).cuda();f.load_state_dict(s['state']);models.append((e,f,s['mean'].cuda(),s['scale'].cuda()))
 dest=A/f'emlp_m112_tasks_{cohort}';dest.mkdir(exist_ok=True);op=V/f'continuous_tasks_{cohort}.json';assert not op.exists();rows=[];intact=[];files=[];ineligible=[];calls=0;replays=0
 for pair in pairs:
  pid=pair['id'];blob=dict(pair_id=pid,writes=[],baselines=[],reconstructions={},intact={},changed={},deltas={});contexts=[]
  for ep in [0,1]:
   ids=torch.tensor([pair['token_ids'][ep]],device='cuda');out,cap=trace(model,ids);calls+=1;base=out[0,-1];assert torch.equal(base.cpu(),native['baselines'][pid+f'|{ep}']['logits']);replays+=1;w=mlp.Down(cap[1]['products']);contexts.append((ids,base,w));blob['writes'].append(w.cpu());blob['baselines'].append(base.cpu())
  recs={}
  for e,f,mu,scale in models:
   for ep,(ids,base,w) in enumerate(contexts):
    rec,support,code=f((w-mu)/scale);q=rec*scale+mu;key=f"{e['seed']}|{e['mode']}|{ep}";recs[key]=q;blob['reconstructions'][key]=dict(write=q.cpu(),support=support.cpu(),code=code.cpu());out,_=trace(model,ids,replace=lambda i,z,y:q+mlp.Down_bias if i==1 else y);calls+=1;out=out[0,-1];blob['intact'][key]=out.cpu();lp=base.double().log_softmax(-1);op_=out.double().log_softmax(-1)
    intact.append(dict(pair_id=pid,family=pair['family'],seed=e['seed'],mode=e['mode'],endpoint=ep,KL=float((lp.exp()*(lp-op_)).sum()),top1_agreement=int(base.argmax()==out.argmax()),margin_error=margin(out,pair['answer_groups'][ep],pair['variants'][ep])-margin(base,pair['answer_groups'][ep],pair['variants'][ep]),local_error=float((q.double()-w.double()).square().sum()),local_energy=float(w.double().square().sum()),active_mean=float(support.sum(-1).mean())))
  targets=[t for t in nr['records'] if t['pair_id']==pid and t['layer']==1];assert len(targets)==6
  for t in targets:
   if t['status']!='evaluated':ineligible.append(dict(pair_id=pid,endpoint=t['endpoint'],region=t['region'],reason=t.get('reason')));continue
   ep=t['endpoint'];region=t['region'];pos=t['recipient_positions'];dp=t['donor_positions'];ids,base,w=contexts[ep];teacher=native['changed'][t['sample_key']].cuda();nd=contexts[1-ep][2][:,dp]-w[:,pos];assert torch.equal(nd.cpu(),native['deltas'][t['sample_key']]['delta']);nlp=base.double().log_softmax(-1);prob=nlp.exp();teacher_response=teacher.double().log_softmax(-1)-nlp
   def patch(delta):
    nonlocal calls
    if not bool(delta.count_nonzero()):return base
    full=torch.zeros_like(w);full[:,pos]=delta;out,_=trace(model,ids,replace=lambda i,z,y:y+full if i==1 else y);calls+=1;return out[0,-1]
   assert torch.equal(patch(nd),teacher);replays+=1
   for seed in [71,72]:
    changes={mode:recs[f'{seed}|{mode}|{1-ep}'][:,dp]-recs[f'{seed}|{mode}|{ep}'][:,pos] for mode in ['hard','continuous']};hn=changes['hard'].double().norm();sn=changes['continuous'].double().norm()
    if float(hn)==0 and float(sn)>0:ineligible.append(dict(pair_id=pid,endpoint=ep,region=region,seed=seed,condition='matched_hard',reason='zero_hard_nonzero_continuous'))
    else:changes['matched_hard']=changes['hard']*(sn/hn).to(changes['hard'].dtype) if float(hn)>0 else changes['hard']
    for condition,delta in changes.items():
     output=patch(delta);response=output.double().log_softmax(-1)-nlp;key=f'{seed}|{ep}|{region}|{condition}';blob['changed'][key]=output.cpu();blob['deltas'][key]=delta.cpu();g=pair['answer_groups'][ep];variant=pair['variants'][ep];tm=margin(teacher,g,variant)-margin(base,g,variant);cm=margin(output,g,variant)-margin(base,g,variant)
     rows.append(dict(pair_id=pid,family=pair['family'],seed=seed,endpoint=ep,region=region,condition=condition,response_error=float((prob*(response-teacher_response).square()).sum()),teacher_response_energy=float((prob*teacher_response.square()).sum()),margin_error=(cm-tm)**2,teacher_margin_energy=tm**2,margin_change=cm,teacher_margin_change=tm,local_error=float((delta.double()-nd.double()).square().sum()),native_delta_energy=float(nd.double().square().sum()),candidate_delta_energy=float(delta.double().square().sum())))
  path=dest/(hashlib.sha256(pid.encode()).hexdigest()+'.pt');assert not path.exists();torch.save(blob,path);files.append(dict(pair_id=pid,path=str(path),sha256=sha(path)));print('COHORT',cohort,'PAIRS',len(files),'FORWARDS',calls,flush=True)
 out=dict(status='completed_pending_audit',cohort=cohort,source_sha256=sha(__file__),execution_sha256=sha(xp),native_sha256=sha(V/(stem+'.json')),models_sha256=sha(V/'continuous_training_results.json'),rows=rows,intact=intact,files=files,ineligible=ineligible,forwards=calls,exact_replays=replays)
 op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',cohort,calls,'ROWS',len(rows),flush=True)
if __name__=='__main__':main()

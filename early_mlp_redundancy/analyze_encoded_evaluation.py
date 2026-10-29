"""M70 all fixed candidates: causal corners, training objective, and retention."""
from pathlib import Path
import json,hashlib,numpy as np,torch
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lp(x):x=x.double().numpy();x=x-x.max();return x-np.log(np.exp(x).sum())
def main():
 torch.set_num_threads(1);models=json.loads((V/'encoded_change_evaluation_models.json').read_text())['models'];results={};bindings={}
 for label,rname,aname in [('holdout','encoded_factorial.json','encoded_factorial_audit.json'),('train','encoded_eval_train.json','encoded_eval_train_audit.json'),('natural','encoded_eval_natural.json','encoded_eval_natural_audit.json')]:
  rp=V/rname;a=json.loads((V/aname).read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);results[label]=json.loads(rp.read_text());bindings[label]=dict(result_sha256=sha(rp),audit_sha256=sha(V/aname))
 protocol=json.loads((V/'balanced_response_training_protocol.json').read_text());den={(x['family'],x['region']):x['denominator'] for x in protocol['scales']};intact={};intactrows=[]
 for split,nname in [('train','training_causal_coverage.json'),('holdout','source_holdout_native.json')]:
  nr=json.loads((V/nname).read_text());sp=Path(nr['samples_path']);assert sha(sp)==nr['samples_sha256'];native=torch.load(sp,weights_only=True)
  for file in results[split]['files']:
   path=Path(file['path']);assert sha(path)==file['sha256'];blob=torch.load(path,weights_only=True)
   for e in models:
    ms=':'.join(str(e[k]) for k in ['kind','seed','arm'])
    for ep in [0,1]:
     key=file['pair_id']+f'|{ep}';b=native['baselines'][key];nl=lp(b['logits']);z=blob['baselines'][ms+f'|{ep}'];sl=lp(z);g=b['groups'];v=b['variant'];lse=lambda x:float(np.logaddexp.reduce(x));margin=lse(sl[g[v]])-lse(sl[g[1-v]]);K=float((np.exp(nl)*(nl-sl)).sum());row={k:e[k] for k in ['kind','seed','arm']}|dict(split=split,key=key,KL=K,binary_correct=margin>0,task_top1=int(z.argmax()) in g[v]);intactrows.append(row);intact[split,e['kind'],e['seed'],e['arm'],key]=K
 rows=[]
 for e in models:
  row={k:e[k] for k in ['kind','seed','arm']}
  for split in ['train','holdout']:
   rr=results[split];summ=[x for x in rr['summaries'] if all(x[k]==e[k] for k in ['kind','seed','arm'])]
   for region in ['changed_input_positions','final_position']:
    site='source' if region=='changed_input_positions' else 'final';gg=[x for x in rr['geometry'] if x['region']==region and all(x[k]==e[k] for k in ['kind','seed','arm'])];row[f'{split}_{site}_local']=sum(x['delta_error'] for x in gg)/sum(x['target_delta_energy'] for x in gg)
    for corner in sorted({x['corner'] for x in summ}):
     ss=[x for x in summ if x['region']==region and x['corner']==corner];pool=next(x for x in ss if x['family'] is None);fs=[x for x in ss if x['family'] is not None and x['relative_response_error'] is not None];prefix=f'{split}_{site}_{corner}';row[prefix+'_pooled']=pool['relative_response_error'];row[prefix+'_family']=sum(x['relative_response_error'] for x in fs)/len(fs);row[prefix+'_margin']=pool['relative_margin_error']
   ir=[x for x in intactrows if x['split']==split and all(x[k]==e[k] for k in ['kind','seed','arm'])]
   for metric in ['KL','binary_correct','task_top1']:row[f'{split}_intact_{metric}']=sum(x[metric] for x in ir)/len(ir)
  nat=next(x for x in results['natural']['summaries'] if all(x[k]==e[k] for k in ['kind','seed','arm']))
  for metric in ['mean_KL','delta_CE','top1_agreement','active_mean','local_relative_error']:row['natural_'+metric]=nat[metric]
  rawlookup={(x['key'],x['region']):x for x in results['train']['rows'] if x['corner']=='native_reconstructed' and all(x[k]==e[k] for k in ['kind','seed','arm'])};scheduled=[]
  for ass in protocol['schedules'][str(e['seed'])]['task_updates']:
   for ep in [0,1]:
    key=ass['pair_id']+f'|{ep}';rr=[rawlookup[key,region] for region in ['changed_input_positions','final_position'] if (key,region) in rawlookup];B=sum(x['response_error']/den[ass['family'],x['region']] for x in rr)/len(rr);scheduled.append((intact['train',e['kind'],e['seed'],e['arm'],key],B))
  row['scheduled_task_K']=sum(x[0] for x in scheduled)/132;row['scheduled_encoded_R']=sum(x[1] for x in scheduled)/132;rows.append(row)
 comparisons=[]
 for control in ['intact_KL','KL_balanced_response']:
  for metric in [k for k in rows[0] if k not in ['kind','seed','arm']]:
   higher=any(w in metric for w in ['binary_correct','task_top1','top1_agreement']);pairs=[]
   for kind in ['scalar','block','shared_fixed']:
    for seed in [71,72]:
     new=next(x for x in rows if (x['kind'],x['seed'],x['arm'])==(kind,seed,'KL_encoded_balanced_response'));old=next(x for x in rows if (x['kind'],x['seed'],x['arm'])==(kind,seed,control));pairs.append(dict(kind=kind,seed=seed,encoded=new[metric],control=old[metric],encoded_better=new[metric]>old[metric] if higher else new[metric]<old[metric]))
   comparisons.append(dict(control=control,metric=metric,encoded_better=sum(x['encoded_better'] for x in pairs),pairs=pairs))
 op=V/'encoded_change_evaluation_comparison.json';assert not op.exists();op.write_text(json.dumps(dict(status='completed',source_sha256=sha(Path(__file__)),bindings=bindings,rows=rows,intact_rows=intactrows,comparisons=comparisons,scope='All18fixed candidates retained; familiar source-group development holdouts. Intact metrics independently derived from saved full final vectors. Family and energy pooling kept distinct; active cost is descriptive, not a performance win.'),indent=2)+'\n');print([(x['control'],x['metric'],x['encoded_better']) for x in comparisons if ('holdout' in x['metric'] and x['metric'].endswith('_family')) or x['metric']=='natural_mean_KL'])
if __name__=='__main__':main()

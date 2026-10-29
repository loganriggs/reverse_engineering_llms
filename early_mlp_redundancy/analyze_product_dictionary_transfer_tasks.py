"""All fixed paired transfer comparisons, preserving scope and aggregation."""
import json,sys
from collections import defaultdict
import numpy as np
import torch
from fit_pca_encoder import sha,V
from audit_pca_factorial import lp,margin

def main():
 torch.set_num_threads(1);cohort=sys.argv[1];assert cohort in ['whole','source'];stem='product_dictionary_factorial' if cohort=='whole' else 'product_dictionary_source_factorial';native_stem='whole_task_native' if cohort=='whole' else 'source_holdout_native';rp=V/(stem+'.json');r=json.loads(rp.read_text());audit=json.loads((V/(stem+'_audit.json')).read_text());assert audit['status']=='passed' and audit['result_sha256']==sha(rp);models=json.loads((V/'product_dictionary_evaluation_models.json').read_text())['models'];pairs={p['id']:p for p in json.loads((V/(native_stem+'_protocol.json')).read_text())['pairs']};native=json.loads((V/(native_stem+'.json')).read_text());assert sha(native['samples_path'])==native['samples_sha256'];saved=torch.load(native['samples_path'],weights_only=True);qualified={x['family'] for x in json.loads((V/'whole_task_native_comparison.json').read_text())['summaries'] if x['competent']} if cohort=='whole' else set();identity=lambda e:tuple(e[k] for k in ['kind','seed','arm'])
 def scopes(family,fmt):return ['all','family:'+family,'format:'+family+':'+fmt]+(['contrast_qualified'] if family in qualified else [])
 intact=defaultdict(list);responses=defaultdict(list);geometry=defaultdict(list)
 for file in r['files']:
  assert sha(file['path'])==file['sha256'];b=torch.load(file['path'],weights_only=True,mmap=True);p=pairs[b['pair_id']];fmt=p.get('format','historical_source_group')
  for e in models:
   mk=identity(e);ms=':'.join(map(str,mk))
   for ep in [0,1]:
    n=saved['baselines'][p['id']+f'|{ep}'];z=b['baselines'][ms+f'|{ep}'];nl=lp(n['logits']);sl=lp(z);row=dict(pair_id=p['id'],endpoint=ep,family=p['family'],KL=float((np.exp(nl)*(nl-sl)).sum()),binary_correct=bool(margin(z,n['groups'],n['variant'])>0),top1_correct=bool(int(z.argmax()) in n['groups'][n['variant']]),native_binary_correct=bool(margin(n['logits'],n['groups'],n['variant'])>0),native_top1_correct=bool(int(n['logits'].argmax()) in n['groups'][n['variant']]),active_mean=float(b['reconstructions'][ms+f'|{ep}']['support'].double().sum(-1).mean()))
    for scope in scopes(p['family'],fmt):intact[mk+(scope,)].append(row)
 for x in r['rows']:
  for scope in scopes(x['family'],x['format']):responses[identity(x)+(scope,x['region'],x['corner'])].append(x)
 for x in r['geometry']:
  for scope in scopes(x['family'],x['format']):geometry[identity(x)+(scope,x['region'])].append(x)
 summaries=[]
 for (kind,seed,arm,scope),rr in intact.items():
  mk=kind,seed,arm;e=next(e for e in models if identity(e)==mk);s=dict(kind=kind,seed=seed,arm=arm,variant=e['variant'],scope=scope,endpoints=len(rr),families=sorted({x['family'] for x in rr}))
  for field in ['KL','binary_correct','top1_correct','native_binary_correct','native_top1_correct','active_mean']:s['intact_'+field]=float(np.mean([x[field] for x in rr]))
  pp=defaultdict(list)
  for x in rr:pp[x['pair_id']].append(x)
  assert all(len(v)==2 for v in pp.values());s['pair_both_correct']=float(np.mean([all(x['binary_correct'] for x in v) for v in pp.values()]));s['native_pair_both_correct']=float(np.mean([all(x['native_binary_correct'] for x in v) for v in pp.values()]))
  for region,site in [('changed_input_positions','source'),('final_position','final')]:
   gg=geometry[mk+(scope,region)];den=sum(x['target_delta_energy'] for x in gg);s[site+'_local']=sum(x['delta_error'] for x in gg)/den if den else None
   for corner in ['reconstructed_native','native_reconstructed','reconstructed_reconstructed']:
    rows=responses[mk+(scope,region,corner)];den=sum(x['target_energy'] for x in rows);md=sum(x['margin_target_energy'] for x in rows);pre=site+'_'+corner;s[pre+'_pooled']=sum(x['response_error'] for x in rows)/den if den else None;s[pre+'_margin']=sum(x['margin_error'] for x in rows)/md if md else None;fr=[]
    for family in s['families']:
     ff=[x for x in rows if x['family']==family];en=sum(x['target_energy'] for x in ff)
     if en:fr.append(sum(x['response_error'] for x in ff)/en)
    s[pre+'_family']=float(np.mean(fr)) if fr else None;s[pre+'_evaluated']=len(rows)
  summaries.append(s)
 lookup={(s['seed'],s['arm'],s['scope']):s for s in summaries};comparisons=[]
 for seed in [71,72]:
  for scope in sorted({s['scope'] for s in summaries}):
   initial=lookup[seed,'initial',scope];kl=lookup[seed,'intact_KL',scope];joint=lookup[seed,'joint_three_corner',scope]
   fields=[k for k in initial if k.endswith(('_family','_pooled','_margin','_local')) or k in ['intact_KL','intact_binary_correct','intact_top1_correct','pair_both_correct']]
   comparisons.append(dict(seed=seed,scope=scope,metrics={k:dict(initial=initial[k],intact_KL=kl[k],joint=joint[k],joint_minus_KL=joint[k]-kl[k],joint_minus_initial=joint[k]-initial[k]) for k in fields if all(x[k] is not None for x in [initial,kl,joint])}))
 out=dict(status='completed',source_sha256=sha(__file__),cohort=cohort,result_sha256=sha(rp),audit_sha256=sha(V/(stem+'_audit.json')),summaries=summaries,comparisons=comparisons,scope='All6models/families/formats retained. Joint vsKLonly withinseed and each versusits initial; no candidate selection. Equalfamily/pooled/margin/local/intact metrics separate. Activecount128outputcoordinates,4608nativeproducts retained.')
 dest=V/(stem+'_comparison.json');assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('COMPARED',cohort,len(summaries),len(comparisons))
if __name__=='__main__':main()

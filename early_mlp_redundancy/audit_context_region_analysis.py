"""Independent scalar-loop accounting for M16 region summaries and role scores."""
from pathlib import Path
from collections import defaultdict
import json,math,hashlib
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(x):return tuple(x[k] for k in ['id','endpoint','kind','seed','grouping','group'])
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)
rp=P/'context_region_roles.json';r=json.loads(rp.read_text());ap=P/'context_region_analysis.json';a=json.loads(ap.read_text());assert sha(rp)==a['records_sha256'];assert sha(P/'context_region_protocol.json')==a['protocol_sha256'];old=json.loads((P/'context_group_roles.json').read_text());matching=json.loads((P/'group_role_matching.json').read_text());byregion={reg:{key(x):x for x in r['records'] if x['region']==reg} for reg in ['prefix','task']};whole={key(x):x for x in old['records']};cohorts=json.loads((P/'context_competence.json').read_text())['cohorts'];matrices={}
for s in a['summary']:
 keys=[k for k,x in whole.items() if all(x[f]==s[f] for f in ['context','kind','grouping','seed'])];assert len(keys)==s['endpoint_group_records'];energy=sum(whole[k]['margin_loss']**2 for k in keys);close(energy,s['whole_margin_energy'])
 for region in ['prefix','task']:
  close(sum(byregion[region][k]['margin_loss']**2 for k in keys)/energy,s[region+'_margin_energy_over_whole']);close(sum(byregion[region][k]['last_KL'] for k in keys)/len(keys),s[region+'_final_KL']);assert sum(byregion[region][k]['noop'] for k in keys)==s[region+'_noops']
 close(sum((whole[k]['margin_loss']-byregion['prefix'][k]['margin_loss']-byregion['task'][k]['margin_loss'])**2 for k in keys)/energy,s['interaction_energy_over_whole']);close(sum(whole[k]['last_KL'] for k in keys)/len(keys),s['whole_final_KL'])
 families=sorted(c['family'] for c in cohorts if c['context']==s['context'] and c['pass_gate'])
 for region in ['prefix','task','interaction']:
  values=defaultdict(list)
  for k in keys:
   x=whole[k];v=x['margin_loss']-byregion['prefix'][k]['margin_loss']-byregion['task'][k]['margin_loss'] if region=='interaction' else byregion[region][k]['margin_loss'];values[x['group'],x['family']].append(v)
  mat=[[sum(values[g,f])/len(values[g,f]) for f in families] for g in range(32)];mk=':'.join(str(s[f]) for f in ['context','kind','grouping','seed'])+':'+region;matrices[mk]=mat
  for row,saved in zip(mat,a['matrices'][mk]):
   for v,t in zip(row,saved):close(v,t)
for score in a['results']:
 match=next(m for m in matching['matches'] if all(m[f]==score[f] for f in ['kind','grouping','method']));assert match['assignment']==score['assignment'];assert match['train_amplitudes']==score['amplitudes'];stem=':'.join(score[f] for f in ['context','kind','grouping']);aa=matrices[stem+':71:'+score['region']];bb=matrices[stem+':72:'+score['region']];energies=[];errors=[];dot=0.;penergy=0.
 for j,f in enumerate(score['families']):
  target=[bb[match['assignment'][g]][j] for g in range(32)];pred=[aa[g][j]*match['train_amplitudes'][g] for g in range(32)];energies.append(sum(v*v for v in target));errors.append(sum((u-v)**2 for u,v in zip(pred,target)));dot+=sum(u*v for u,v in zip(pred,target));penergy+=sum(v*v for v in pred)
 out=score['score'];close(sum(errors)/sum(energies),out['relative_squared_error']);close(dot/max(math.sqrt(penergy)*math.sqrt(sum(energies)),1e-20),out['flattened_cosine']);close(sum(e/v for e,v in zip(errors,energies))/len(energies),out['equal_family_mean_relative_error']);keep=[i for i,f in enumerate(score['families']) if f!='antonym'];close(sum(errors[i] for i in keep)/sum(energies[i] for i in keep),out['relative_error_without_antonym'])
 for f,e,v in zip(score['families'],errors,energies):close(e/v,out['per_family_errors'][f]);close(v/sum(energies),out['family_energy_shares'][f])
out=dict(status='passed',region_summaries=len(a['summary']),family_matrices=len(matrices),frozen_scores=len(a['results']),analysis_sha256=sha(ap),scope='Independent scalar-loop energy/KL/interaction, family means and frozen prediction metrics, no fitting.')
(P/'context_region_analysis_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

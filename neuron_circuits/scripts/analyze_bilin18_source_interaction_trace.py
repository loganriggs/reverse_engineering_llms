from pathlib import Path
import json,math
from collections import defaultdict
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_source_interaction_trace';j=json.loads((A/f'{S}.json').read_text());groups=defaultdict(list)
for r in j['records']:groups[r['format'],r['kind'],r['layer'],r['stage'],r['position']].append(r)
rows=[]
for (f,kind,l,stage,pos),rr in groups.items():
 n=sum(r['cases'] for r in rr);I=sum(r['interaction_squared_norm'] for r in rr);D=sum(r['actual_joint_change_squared_norm'] for r in rr);N=sum(r['native_squared_norm'] for r in rr);rows.append({'format':f,'kind':kind,'layer':l,'stage':stage,'position':pos,'interaction_vector_RMS':math.sqrt(I/n),'actual_joint_change_vector_RMS':math.sqrt(D/n),'interaction_relative_to_joint_change_RMS':math.sqrt(I/D) if D/n>1e-10 else None,'interaction_relative_to_native_RMS':math.sqrt(I/N) if N else None})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':rows,'limitations':'Vector norms are stage-local descriptive geometry; not causal attribution, necessity or sufficiency.'},indent=2)+'\n');print(j['audits'])
for f in [0,1]:
 print('FORMAT',f,'coherent query stages')
 for r in rows:
  if r['format']==f and r['kind']=='coherent' and r['position']=='query' and (r['layer']<=3 or r['stage']=='block_output'):print(r['layer'],r['stage'],round(r['interaction_vector_RMS'],6),round(r['actual_joint_change_vector_RMS'],6),r['interaction_relative_to_joint_change_RMS'])
 print('second source L2')
 for r in rows:
  if r['format']==f and r['kind']=='coherent' and r['position']=='second_source' and r['layer']==2:print(r['stage'],r['interaction_vector_RMS'],r['interaction_relative_to_joint_change_RMS'])

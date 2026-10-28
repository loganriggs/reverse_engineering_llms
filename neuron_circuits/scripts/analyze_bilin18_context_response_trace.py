from pathlib import Path
import torch,json,numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_context_response_trace'
torch.set_num_threads(6)
c=torch.load(A/f'{S}_captures.pt',weights_only=False);ex=c['examples'];tr=c['traces'];lookup={(e['format'],e['item'],e['query'],e['same_key']):rid for rid,e in ex.items()};rows=[];control=[]
for rid,e in ex.items():
 if e['same_key']:continue
 other=lookup[e['format'],e['item'],e['query'],True]
 for state in [False,True]:
  for layer,stage in tr[rid,state,0.]:
   a=(tr[rid,state,.5][layer,stage]-tr[rid,state,0.][layer,stage]).double()
   b=(tr[other,state,.5][layer,stage]-tr[other,state,0.][layer,stage]).double()
   for pos,role in enumerate(['source1','source2','query']):
    x,y=a[pos],b[pos];nx=float(x.norm());ny=float(y.norm());diff=float((x-y).norm());dot=float(x@y)
    rows.append(dict(item=e['item'],format=e['format'],query=e['query'],state_same_key=state,layer=layer,stage=stage,role=role,norm_different=nx,norm_same=ny,difference_norm=diff,dot=dot))
    if role=='source1':control.extend([nx,ny])
summary=[]
for f in [0,1]:
 for layer in range(2,18):
  for stage in ['attention','mlp','residual']:
   for role in ['source2','query']:
    rr=[r for r in rows if r['format']==f and r['layer']==layer and r['stage']==stage and r['role']==role]
    x=np.mean([r['norm_different']**2 for r in rr]);y=np.mean([r['norm_same']**2 for r in rr]);d=np.mean([r['difference_norm']**2 for r in rr]);dot=np.mean([r['dot'] for r in rr]);summary.append(dict(format=f,layer=layer,stage=stage,role=role,response_RMS_different=float(np.sqrt(x)),response_RMS_same=float(np.sqrt(y)),context_difference_RMS=float(np.sqrt(d)),relative_context_difference=float(np.sqrt(d/max((x+y)/2,1e-30))),pooled_cosine=float(dot/max(np.sqrt(x*y),1e-30))))
audit={'source1_response_max_norm':max(control),'all_finite':all(np.isfinite(r['difference_norm']) for r in rows)}
assert audit['source1_response_max_norm']<1e-5 and audit['all_finite']
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'records':rows,'audits':audit,'limitations':'Pooled across local states and query choices; shared lexical items, descriptive. Norms are not causal importance and later stages have different scale. Zero-response cosine is undefined and encoded as zero.'},indent=2)+'\n')
print(audit)
for r in summary:
 if r['layer'] in [2,3,4,8,17] and (r['stage']=='attention' or r['stage']=='residual'):print(r)

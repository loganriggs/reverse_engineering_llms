import json,torch,hashlib
from pathlib import Path
from run_bilin18_diverse_causal_tasks import ROOT,ART
S='bilin18_operation_control_panel';rows=json.loads((ART/f'{S}_manifest.json').read_text());cap=torch.load(ART/f'{S}_captures.pt',weights_only=False);Y=torch.cat([cap['normalized_mlp_contexts'][l].double() for l in range(10,18)],dim=-1);lookup={(r['family'],r['control'],r['item'],r['form']):i for i,r in enumerate(rows)};families=sorted({r['family'] for r in rows});shifts={};means={}
# Two contexts per axis: verb at each suffix, suffix at each verb. No pooling before fitting.
for family in families:
 for control in ['task','copy','constant']:
  for axis,pairs in [('verb',[(0,1),(2,3)]),('suffix',[(0,2),(1,3)])]:
   for setting,(a,b) in enumerate(pairs):
    key=family,control,axis,setting;D=torch.stack([Y[lookup[family,control,i,b]]-Y[lookup[family,control,i,a]] for i in range(8)]);shifts[key]=D;means[key]=D[:4].mean(0)
records=[]
for family in families:
 for axis in ['verb','suffix']:
  for setting in range(2):
   key=family,'task',axis,setting;D=shifts[key][4:];own=means[key]
   controls={c:means[family,c,axis,setting] for c in ['copy','constant']};controls['own']=own;controls['other_tasks']=torch.stack([means[f,'task',axis,setting] for f in families if f!=family]).mean(0)
   for control,pred in controls.items():
    for item,d in enumerate(D,4):records.append(dict(family=family,axis=axis,setting=setting,item=item,predictor=control,error=float((d-pred).norm()/d.norm()),cosine=float(torch.nn.functional.cosine_similarity(d,pred,dim=0)),target_norm=float(d.norm()),prediction_norm=float(pred.norm())))
a={'status':'complete','design':'Discovery first4 items per family/control/context; heldout last4. Predict concatenated normalized MLP10-17 context shift. Own-task, same-input copy/constant controls, leave-family-out task mean. No heldout fitting; geometry only, not output fidelity.','records':records,'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ART/f'{S}_manifest.json',ART/f'{S}_captures.pt',Path(__file__)]}};(ART/'bilin18_operation_control_geometry.json').write_text(json.dumps(a,indent=2));print('COMPLETE',len(records))

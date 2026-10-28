"""Post-result descriptive class specificity on clean inputs; no fitted assignment."""
import json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation';rows=[]
for width,receipt,sample in [(8,'capacity_toys','/tmp/emlp_t2_models.pt'),(16,'width_toys','/tmp/emlp_t3_models.pt')]:
 r=json.loads((P/(receipt+'.json')).read_text());saved=torch.load(sample,weights_only=True)
 for s in saved:
  row=r['rows'][s['row']]
  if row['classes']!=16:continue
  st={k:v.double().numpy() for k,v in s['state'].items()};x=s['clean_inputs'].double().numpy();labels=s['labels'].numpy();h=(x@st['L'].T)*(x@st['R'].T);e=np.square(h)*np.square(st['D']).sum(0)
  perclass=np.stack([e[labels==c].mean(0) for c in range(16)])
  specificity=float(perclass.max(0).sum()/perclass.sum());assert 1/16-1e-12<=specificity<=1+1e-12
  rows.append(dict(width=width,row=s['row'],train_noise=row['train_noise'],penalty=row['penalty'],seed=row['seed'],class_energy_specificity=specificity,distinct_dominant_channels=int(len(set(perclass.argmax(1).tolist())))))
out=dict(protocol='Post-result descriptive statistic. For balanced clean classes, sum_over_channels max_class(mean contribution energy) divided by total class/channel mean energy. Range1/16(uniform across classes) to1(each channel active in just one class). No learned alignment or threshold. Not a causal necessity measure.',rows=rows)
(P/'toy_specialization.json').write_text(json.dumps(out,indent=2)+'\n')
for w in (8,16):
 for n in (0,.25):
  for p in (0,.01):
   rr=[r for r in rows if (r['width'],r['train_noise'],r['penalty'])==(w,n,p)]
   print(w,n,p,'specificity',np.mean([r['class_energy_specificity'] for r in rr]),'distinct',np.mean([r['distinct_dominant_channels'] for r in rr]))

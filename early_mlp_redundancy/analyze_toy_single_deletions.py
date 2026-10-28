"""Post-result exhaustive single-channel stress test on fixed width16 toys."""
import json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation';saved=torch.load('/tmp/emlp_t3_models.pt',weights_only=True);receipt=json.loads((P/'width_toys.json').read_text());rows=[]
for s in saved:
 r=receipt['rows'][s['row']];st={k:v.double().numpy() for k,v in s['state'].items()};x=s['clean_inputs'].double().numpy();labels=s['labels'].numpy();centers=s['centers'].double().numpy();h=(x@st['L'].T)*(x@st['R'].T);pred=h@st['D'].T+st['b'];matrix=[];confusions=[]
 for j in range(16):
  edited=pred-h[:,j,None]*st['D'][:,j][None,:];guess=(edited@centers.T).argmax(-1)
  matrix.append([float((guess[labels==c]==c).mean()) for c in range(16)])
  confusions.append(np.bincount(labels*16+guess,minlength=256).reshape(16,16).tolist())
 a=np.array(matrix)
 rows.append(dict(row=s['row'],train_noise=r['train_noise'],penalty=r['penalty'],seed=r['seed'],channel_by_class_accuracy=matrix,confusions=confusions,mean_accuracy_over_single_deletions=float(a.mean()),worst_channel_global_accuracy=float(a.mean(1).min()),mean_class_accuracy_under_its_worst_channel=float(a.min(0).mean()),classes_with_any_single_deletion_error=int((a.min(0)<1).sum())))
(P/'toy_single_deletions.json').write_text(json.dumps(dict(protocol='Post-result exhaustive white-box single-channel deletions on all12 fixed width16 models, independent NumPy forward on clean draws. Worst channel may differ for each class; that aggregate is not one common intervention or a held-out attack. No training or selection of models.',rows=rows),indent=2)+'\n')
for n in (0,.25):
 for p in (0,.01):
  rr=[r for r in rows if r['train_noise']==n and r['penalty']==p]
  print(n,p,{k:np.mean([r[k] for r in rr]) for k in ['mean_accuracy_over_single_deletions','worst_channel_global_accuracy','mean_class_accuracy_under_its_worst_channel','classes_with_any_single_deletion_error']})

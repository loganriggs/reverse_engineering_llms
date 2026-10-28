"""Torch replay of all192 NumPy single-channel confusion matrices."""
import json,torch
from pathlib import Path
P=Path(__file__).resolve().parent/'validation';r=json.loads((P/'toy_single_deletions.json').read_text());saved=torch.load('/tmp/emlp_t3_models.pt',weights_only=True);mismatch=0;maxdiff=0
for s in saved:
 st=s['state'];h=(s['clean_inputs']@st['L'].T)*(s['clean_inputs']@st['R'].T)
 for j in range(16):
  hh=h.clone();hh[:,j]=0;v=hh@st['D'].T+st['b'];pred=(v@s['centers'].T).argmax(-1);conf=torch.bincount(s['labels']*16+pred,minlength=256).reshape(16,16);ref=torch.tensor(r['rows'][s['row']]['confusions'][j]);mismatch+=int((conf!=ref).any());maxdiff=max(maxdiff,int((conf-ref).abs().max()))
out=dict(torch_vs_numpy_confusion_mismatch_cases=mismatch,max_confusion_cell_difference=maxdiff,cases=192);assert mismatch==0
(P/'toy_single_deletions_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

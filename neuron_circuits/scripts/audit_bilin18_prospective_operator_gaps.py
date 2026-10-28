import json,hashlib
from pathlib import Path
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ART
@torch.no_grad()
def main():
 torch.set_num_threads(6);paths={'manifest':ART/'bilin18_prospective_operator_panel_manifest.json','capture':ART/'bilin18_prospective_operator_panel_captures.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());s=torch.load(paths['capture'],weights_only=False);lookup={(e['family'],e['form'],e['item']):i for i,e in enumerate(rows)};records=[];hamming=[]
 for i,e in enumerate(rows):
  p=s['native_logits'][i].double().softmax(0)
  def norm(z):return ((z-(p*z).sum()).square()*p).sum().sqrt()
  for kind in ['cross','within']:
   jj=lookup[e['family'],e['form']^1,e['item']] if kind=='cross' else lookup[e['family'],e['form'],(e['item']+4)%8]
   if kind=='cross' and e['form']<2:hamming.append(sum(a!=b for a,b in zip(e['ids'],rows[jj]['ids'])))
   for name,g in s['responses'].items():records.append(dict(id=e['id'],family=e['family'],form=e['form'],contrast=e['contrast'],kind=kind,probe=name,gap_norm=float(norm(g[jj]-g[i])),native_norm=float(norm(g[i])),relative_gap=float(norm(g[jj]-g[i])/norm(g[i]))))
 result=dict(records=records,matched_token_hamming=sorted(set(hamming)),hashes=hashes,hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()});(ART/'bilin18_prospective_operator_gap_audit.json').write_text(json.dumps(result,indent=2))
 for f in ['copy','fact']:
  for con in ['matched','broad']:
   print(f,con,[(kind,round(np.mean([r['gap_norm'] for r in records if r['family']==f and r['contrast']==con and r['kind']==kind]),4),round(np.mean([r['relative_gap'] for r in records if r['family']==f and r['contrast']==con and r['kind']==kind]),3)) for kind in ['cross','within']])
if __name__=='__main__':main()

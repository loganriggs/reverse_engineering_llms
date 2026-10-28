import json,hashlib
from pathlib import Path
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ART
@torch.no_grad()
def main():
 torch.set_num_threads(6);stem='bilin18_matched_mlp5_contexts';paths={'capture':ART/f'{stem}_captures.pt','source':ART/'bilin18_cross_task_response_panel_captures.pt','manifest':ART/'bilin18_cross_task_response_panel_manifest.json','runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};c=torch.load(paths['capture'],weights_only=False);s=torch.load(paths['source'],weights_only=False);rows=json.loads(paths['manifest'].read_text());lookup={(e['family'],e['form'],e['item']):i for i,e in enumerate(rows)};records=[]
 for name,outputs in c['outputs'].items():
  for i,e in enumerate(rows):
   p=s['native_logits'][i].double().softmax(0)
   def center(x):return x-(p*x).sum()
   def dot(x,y):return (x*y*p).sum()
   for kind in ['cross','within']:
    donor=lookup[e['family'],1-e['form'],e['item']] if kind=='cross' else lookup[e['family'],e['form'],(e['item']+4)%8];gap=center(s['responses'][name][donor]-outputs['native'][i]);gn=dot(gap,gap).sqrt()
    for mode in ['context','norm','both']:
     change=center(outputs[f'{kind}_{mode}'][i]-outputs['native'][i]);res=gap-change;cn=dot(change,change).sqrt();records.append(dict(id=e['id'],family=e['family'],form=e['form'],probe=name,kind=kind,mode=mode,gap_norm=float(gn),remaining_relative_gap=float(dot(res,res).sqrt()/gn),cosine=float(dot(change,gap)/(cn*gn)),signed_parallel_fraction=float(dot(change,gap)/gn.square())))
 result=dict(records=records,hashes=hashes,hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},finite=all(np.isfinite(r['remaining_relative_gap']) for r in records));(ART/f'{stem}_gap_audit.json').write_text(json.dumps(result,indent=2))
 for f in sorted({r['family'] for r in records}):
  print(f,[(kind,mode,round(np.mean([r['remaining_relative_gap'] for r in records if r['family']==f and r['kind']==kind and r['mode']==mode]),3),round(np.mean([r['cosine'] for r in records if r['family']==f and r['kind']==kind and r['mode']==mode]),3)) for kind in ['cross','within'] for mode in ['context','both']])
if __name__=='__main__':main()

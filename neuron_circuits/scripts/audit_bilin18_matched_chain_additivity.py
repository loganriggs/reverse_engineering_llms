import json,hashlib
from pathlib import Path
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ART
@torch.no_grad()
def main():
 torch.set_num_threads(6);stem='bilin18_matched_early_context_chain';paths={'chain':ART/f'{stem}_captures.pt','first':ART/'bilin18_matched_mlp5_contexts_captures.pt','source':ART/'bilin18_cross_task_response_panel_captures.pt','manifest':ART/'bilin18_cross_task_response_panel_manifest.json','runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};c=torch.load(paths['chain'],weights_only=False)['outputs'];f=torch.load(paths['first'],weights_only=False)['outputs'];s=torch.load(paths['source'],weights_only=False);rows=json.loads(paths['manifest'].read_text());lookup={(e['family'],e['form'],e['item']):i for i,e in enumerate(rows)};records=[]
 for name,outs in c.items():
  for i,e in enumerate(rows):
   p=s['native_logits'][i].double().softmax(0)
   def center(x):return x-(p*x).sum()
   def norm(x):return (p*center(x).square()).sum().sqrt()
   for kind in ['cross','within']:
    jj=lookup[e['family'],1-e['form'],e['item']] if kind=='cross' else lookup[e['family'],e['form'],(e['item']+4)%8];g=outs['native'][i];delta5=f[name][kind+'_context'][i]-f[name]['native'][i];delta69=outs[kind+'_6_9'][i]-g;joint=outs[kind+'_5_9'][i];additive=g+delta5+delta69;donor=s['responses'][name][jj];gap=norm(donor-g);records.append(dict(id=e['id'],family=e['family'],form=e['form'],probe=name,kind=kind,additive_remaining_gap=float(norm(donor-additive)/gap),joint_remaining_gap=float(norm(donor-joint)/gap),interaction_over_gap=float(norm(joint-additive)/gap),interaction_over_joint_change=float(norm(joint-additive)/norm(joint-g))))
 d=dict(records=records,hashes=hashes,hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},finite=all(np.isfinite(r['additive_remaining_gap']) for r in records));(ART/f'{stem}_additivity_audit.json').write_text(json.dumps(d,indent=2))
 for family in sorted({r['family'] for r in records}):
  rr=[r for r in records if r['family']==family and r['kind']=='cross'];print(family,{k:round(np.mean([r[k] for r in rr]),3) for k in ['additive_remaining_gap','joint_remaining_gap','interaction_over_gap','interaction_over_joint_change']})
if __name__=='__main__':main()

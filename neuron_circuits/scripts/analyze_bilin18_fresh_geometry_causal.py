import json
from pathlib import Path
import numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_fresh_geometry_causal'
j=json.loads((A/f'{S}.json').read_text());rows=j['records'];summary=[];contrasts=[];rng=np.random.default_rng(825)
for site in ['residual0','residual1']:
 for mode in ['block','pca','random','native_block']:
  for arm in ['near','far']:
   rr=[r for r in rows if r['site']==site and r['mode']==mode and r['arm']==arm];summary.append({'site':site,'mode':mode,'arm':arm,'n':len(rr),'mean_receiver_KL':float(np.mean([r['receiver_kl'] for r in rr])),'median_receiver_KL':float(np.median([r['receiver_kl'] for r in rr])),'mean_donor_gain':float(np.mean([r['donor_kl_gain'] for r in rr])),'median_donor_gain':float(np.median([r['donor_kl_gain'] for r in rr])),'positive_donor_gain':sum(r['donor_kl_gain']>0 for r in rr)})
 def contrast(name,rr,diff):
  vals=np.array([np.mean([diff(r) for r in rr if r['block']==b]) for b in sorted({r['block'] for r in rr})]);boot=vals[rng.integers(len(vals),size=(4096,len(vals)))].mean(1);contrasts.append({'site':site,'contrast':name,'blocks':len(vals),'mean':float(vals.mean()),'descriptive_block_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'per_block':vals.tolist()})
 idx={(r['pair'],r['arm'],r['mode']):r for r in rows}
 for mode in ['block','pca','random']:
  rr=[r for r in rows if r['site']==site and r['mode']==mode and r['arm']=='near'];contrast(mode+' far-minus-near receiver KL',rr,lambda r:idx[r['pair'],'far',mode]['receiver_kl']-r['receiver_kl'])
 for arm in ['near','far']:
  rr=[r for r in rows if r['site']==site and r['mode']=='block' and r['arm']==arm]
  for ctrl in ['pca','random']:contrast(arm+' block-minus-'+ctrl+' donor gain',rr,lambda r:r['donor_kl_gain']-idx[r['pair'],arm,ctrl]['donor_kl_gain'])
out={'summary':summary,'contrasts':contrasts,'limitations':'Block-bootstrap descriptive only;6/8 blocks and possible shared documents across blocks. No multiplicity correction. Near/far have different donor distributions; interpret gain relative matched controls.'};(A/f'{S}_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

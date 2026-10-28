"""Post-result M15 error concentration: descriptive accounting, no fitting or selection."""
from pathlib import Path
import json, hashlib
import numpy as np
P=Path(__file__).resolve().parent/'validation'
def main():
 source=P/'context_role_transfer.json';d=json.loads(source.read_text());oldpath=P/'group_role_matching.json';old=json.loads(oldpath.read_text());rows=[]
 for r in d['results']:
  if r['kind']!='block' or r['grouping']!='contiguous':continue
  key=r['context']+':block:contiguous:';a=np.array(d['matrices'][key+'71']);b=np.array(d['matrices'][key+'72'])[r['assignment']];alpha=np.array(r['amplitudes']);pred=a*alpha[:,None];e=((pred-b)**2).sum(1);i=int(e.argmax())
  ta=np.array(old['signatures']['block:contiguous:71:train']['margin'])[i];tb=np.array(old['signatures']['block:contiguous:72:train']['margin'])[r['assignment'][i]]
  expected=max(0.,float(ta@tb/max(ta@ta,1e-20)));assert np.isclose(expected,alpha[i]);assert np.isclose(e.sum()/(b*b).sum(),r['current']['relative_squared_error'])
  rows.append(dict(context=r['context'],method=r['method'],largest_error_source_group=i,target_group=r['assignment'][i],frozen_amplitude=float(alpha[i]),error_share=float(e[i]/e.sum()),predicted_to_target_energy=float((pred*pred).sum()/(b*b).sum()),training_source_norm=float(np.linalg.norm(ta)),training_target_norm=float(np.linalg.norm(tb)),current_source_norm=float(np.linalg.norm(a[i])),current_target_norm=float(np.linalg.norm(b[i]))))
 out=dict(scope='Post-result descriptive concentration; no refit, clipping, group omission or new confirmation.',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),matching_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),rows=rows)
 (P/'context_role_concentration.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()

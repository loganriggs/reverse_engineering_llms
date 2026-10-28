"""H748: effect-size stratification of frozen edits; no refitting or exclusions by effect."""
import json,hashlib,math
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('/home/loganriggs/Coding/neuron_circuits');ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM='bilin18_operation_sign_margins'
def main():
 path=ART/'bilin18_fresh_wording_subspace_effects.json';digest=hashlib.sha256(path.read_bytes()).hexdigest();cfg={'experiment':'H748','input_sha256':digest,'design':'Posthoc H747 sign-margin diagnostic. No fitting or interventions. All contexts retained; eligible sign cases require native*full<0. Native competence reported separately. Fixed absolute target-effect bins and relative target/edit bins, plus continuous failure distributions. No threshold chosen to optimize results.','absolute_thresholds':[0,.01,.05,.1,.25,.5],'relative_thresholds':[0,.01,.05,.1,.25,.5]};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 d=json.loads(path.read_text());by=defaultdict(dict)
 for r in d['records']:by[r['id'],r['receiver']][r['scope']]=r
 records=[]
 for key,v in by.items():
  r=v['native'];cell='|'.join(r[k] for k in ['name_set','wording','format']);competent=all(d['competence'][cell+'|'+f]['pass'] for f in ['entity_binding','sequence_copy']);n=v['native']['absent_identity_effect'];f=v['full']['absent_identity_effect']
  for scope in ['frozen_H742','operation_new','operation_pooled','identity','wording']:
   p=v[scope]['absent_identity_effect'];records.append({'id':key[0],'receiver':key[1],'cell':cell,'family':r['family'],'scope':scope,'competent':competent,'native':n,'full':f,'predicted':p,'eligible':n*f<0,'success':p*f>0,'target_magnitude':abs(f),'edit_magnitude':abs(f-n),'relative_target_margin':abs(f)/max(abs(f-n),1e-15),'absolute_error':abs(p-f)})
 summaries=[]
 for competent in [True,False]:
  for scope in ['frozen_H742','operation_new','operation_pooled','identity','wording']:
   rr=[r for r in records if r['competent']==competent and r['scope']==scope];eligible=[r for r in rr if r['eligible']];s={'competent':competent,'scope':scope,'contexts':len(rr),'eligible':len(eligible),'success':sum(r['success'] for r in eligible),'bins':{},'outcomes':{}}
   for metric,thresholds in [('target_magnitude',cfg['absolute_thresholds']),('relative_target_margin',cfg['relative_thresholds'])]:
    s['bins'][metric]=[{'minimum':t,'success':sum(r['success'] for r in eligible if r[metric]>=t),'total':sum(r[metric]>=t for r in eligible)} for t in thresholds]
   for outcome in [True,False]:
    rows=[r for r in eligible if r['success']==outcome];s['outcomes'][str(outcome)]={'count':len(rows),'families':{f:sum(r['family']==f for r in rows) for f in ['entity_binding','sequence_copy']},'distribution':{m:{'min':float(min(r[m] for r in rows)),'median':float(np.median([r[m] for r in rows])),'max':float(max(r[m] for r in rows))} if rows else None for m in ['target_magnitude','relative_target_margin','absolute_error']}}
   summaries.append(s)
 audits={'hash_unchanged':hashlib.sha256(path.read_bytes()).hexdigest()==digest,'contexts':len(by),'finite':all(math.isfinite(r['relative_target_margin']) and math.isfinite(r['absolute_error']) for r in records),'whole_model_forwards':0,'test_evaluations':0};assert audits['hash_unchanged'] and audits['finite'] and len(by)==240
 (ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'summary':summaries,'audits':audits},indent=2)+'\n')
 for s in summaries:
  if s['competent'] and s['scope'] in ['frozen_H742','operation_pooled']:print(json.dumps(s,indent=2))
 print(audits)
if __name__=='__main__':main()

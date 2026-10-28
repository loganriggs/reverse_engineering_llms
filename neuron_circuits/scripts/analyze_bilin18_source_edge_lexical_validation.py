from pathlib import Path
import json,math
import numpy as np
import torch
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_source_edge_lexical_validation';j=json.loads((A/f'{S}.json').read_text());c=torch.load(A/f'{S}_captures.pt',weights_only=False);idx={(r['id'],r['a'],r['b'],r['arm']):np.array(r['candidate_logit_changes']) for r in j['records']};items=[]
for rid in sorted(c['states']):
 r=next(r for r in j['records'] if r['id']==rid);margin=lambda v:float(v[1]-v[0]);I={arm:margin(idx[rid,1,1,arm]-idx[rid,1,0,arm]-idx[rid,0,1,arm]+idx[rid,0,0,arm]) for arm in j['config']['arms']};effects={arm:margin(idx[rid,1,1,arm]-idx[rid,1,1,'none']) for arm in j['config']['arms']};pat=c['states'][rid]['pattern'].double().reshape(2,2,9)[:,:,5];f=c['states'][rid]['factors'];scores=[]
 for qn,kn in [('q','k'),('q2','k2')]:
  q=f[qn].double().reshape(2,2,9,128)[:,:,5];k=f[kn].double().reshape(2,2,9,128)[:,:,5];scores.append((q*k).sum(-1)/128)
 items.append({'id':rid,'item':r['item'],'format':r['format'],'interaction':I,'removal_effect':effects,'head5_signed_pattern_matching_mean':float((pat[0,0]+pat[1,1])/2),'head5_signed_pattern_mismatch_mean':float((pat[0,1]+pat[1,0])/2),'head5_signed_pattern_matching_minus_mismatch':float((pat[0,0]+pat[1,1]-pat[0,1]-pat[1,0])/2),'factor1_matching_minus_mismatch':float((scores[0][0,0]+scores[0][1,1]-scores[0][0,1]-scores[0][1,0])/2),'factor2_matching_minus_mismatch':float((scores[1][0,0]+scores[1][1,1]-scores[1][0,1]-scores[1][1,0])/2)})
summary=[];rng=np.random.default_rng(796)
for fmt in [0,1]:
 rr=[r for r in items if r['format']==fmt];den=sum(r['removal_effect']['all_edges']**2 for r in rr);row={'format':fmt,'head5_vs_all_removal_margin_relative_RMS_error':math.sqrt(sum((r['removal_effect']['head5']-r['removal_effect']['all_edges'])**2 for r in rr)/den),'all_edge_removal_margin_RMS':math.sqrt(den/len(rr)),'remaining_interaction_RMS_ratio':{arm:math.sqrt(sum(r['interaction'][arm]**2 for r in rr)/sum(r['interaction']['none']**2 for r in rr)) for arm in j['config']['arms']},'removal_effect_mean':{arm:float(np.mean([r['removal_effect'][arm] for r in rr])) for arm in j['config']['arms']}}
 for metric in ['head5_signed_pattern_matching_mean','head5_signed_pattern_mismatch_mean','head5_signed_pattern_matching_minus_mismatch','factor1_matching_minus_mismatch','factor2_matching_minus_mismatch']:
  v=np.array([r[metric] for r in rr]);boot=v[rng.integers(len(v),size=(4096,len(v)))].mean(1);row[metric]={'mean':float(v.mean()),'descriptive_item_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'positive_items':int(sum(v>0))}
 summary.append(row)
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'items':items,'limitations':'Pairwise margin validation, not full16-candidate specificity. Frozen head but previously studied task panel. Two partially substituted identities per item do not distinguish semantic equality from geometric similarity. Item bootstrap descriptive.'},indent=2)+'\n');print(json.dumps(summary,indent=2))

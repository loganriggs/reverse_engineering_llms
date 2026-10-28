from pathlib import Path
import json
import numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_source_donor_specificity';j=json.loads((A/f'{S}.json').read_text());rr=[r for r in j['records'] if r['donor_variant']!=0];metrics=['receiver_logprob_change','donor_logprob_change','donor_vs_receiver_margin_change','actual_donor_vs_other_alternatives_change','reference_alternate_vs_receiver_margin_change','donor_vs_reference_alternate_change'];rng=np.random.default_rng(791);summary=[]
for f in [0,1]:
 for site in j['config']['sites']:
  for donor_set in ['all','reference','alternative_controls']:
   rows=[r for r in rr if r['format']==f and r['site']==site and (donor_set=='all' or (donor_set=='reference' and r['donor_variant']==1) or (donor_set=='alternative_controls' and r['donor_variant'] in [2,3]))];row={'format':f,'site':site,'donor_set':donor_set,'interventions':len(rows)}
   for metric in metrics:
    v=np.array([np.mean([r[metric] for r in rows if r['item']==item]) for item in sorted({r['item'] for r in rows})]);boot=v[rng.integers(len(v),size=(4096,len(v)))].mean(1);row[metric]={'mean':float(v.mean()),'descriptive_item_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}
   ranks=[]
   for r in rows:
    a=j['candidate_words'].index(r['receiver_word']);b=j['candidate_words'].index(r['donor_word']);dl=r['candidate_logit_changes'];ranks.append(1+sum(dl[z]>dl[b] for z in range(len(dl)) if z!=a and z!=b))
   row['actual_donor_largest_nonreceiver_logit_increase_rate']=float(np.mean(np.array(ranks)==1));row['candidate_prediction_switch_rate']=float(np.mean([r['candidate_prediction']!=r['receiver_word'] for r in rows]));row['donor_full_vocab_top1_rate']=float(np.mean([r['donor_full_vocab_top1'] for r in rows]));summary.append(row)
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'limitations':'16 item clusters per format with donors averaged within item; shared words across items and shared items across formats; descriptive bootstrap without multiplicity correction.'},indent=2)+'\n')
for r in summary:
 if r['donor_set']=='all':print(r['format'],r['site'],{m:round(r[m]['mean'],5) for m in metrics},'donor_rank1',r['actual_donor_largest_nonreceiver_logit_increase_rate'],'flips',r['candidate_prediction_switch_rate']);print('specificity CI',r['actual_donor_vs_other_alternatives_change']['descriptive_item_bootstrap_95'])
for r in summary:
 if r['donor_set']=='alternative_controls':print('alternative',r['format'],r['site'],'actual_vs_ref',r['donor_vs_reference_alternate_change'],'ref_vs_receiver',r['reference_alternate_vs_receiver_margin_change'])

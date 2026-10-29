"""M72 retain all dictionary/encoder variants and matched causal/retention metrics."""
from pathlib import Path
import json,torch,numpy as np
from audit_support_change_diagnostic import sha,lp,margin
P=Path(__file__).resolve().parent;V=P/'validation'
def main():
 torch.set_num_threads(1);fp=V/'least_squares_factorial.json';np_=V/'least_squares_natural.json';f=json.loads(fp.read_text());n=json.loads(np_.read_text())
 for rp,ap in [(fp,V/'least_squares_factorial_audit.json'),(np_,V/'least_squares_natural_audit.json')]:a=json.loads(ap.read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp)
 models=json.loads((V/'least_squares_evaluation_models.json').read_text())['models'];native=json.loads((V/'source_holdout_native.json').read_text());sp=Path(native['samples_path']);assert sha(sp)==native['samples_sha256'];saved=torch.load(sp,weights_only=True);intacts=[]
 for file in f['files']:
  p=Path(file['path']);assert sha(p)==file['sha256'];b=torch.load(p,weights_only=True)
  for e in models:
   ms=':'.join(str(e[k]) for k in ['kind','seed','arm'])
   for ep in [0,1]:
    key=file['pair_id']+f'|{ep}';base=saved['baselines'][key];nl=lp(base['logits']);z=b['baselines'][ms+f'|{ep}'];sl=lp(z);g=base['groups'];v=base['variant'];rec=b['reconstructions'][ms+f'|{ep}'];intacts.append({k:e[k] for k in ['kind','seed','arm']}|dict(key=key,KL=float((np.exp(nl)*(nl-sl)).sum()),binary_correct=bool(margin(sl,g,v)>0),task_top1=int(z.argmax()) in g[v],selected_mean=float(rec['support'].double().sum(-1).mean()),nonzero_mean=float((rec['code']!=0).double().sum(-1).mean())))
 rows=[]
 for e in models:
  out={k:e[k] for k in ['kind','seed','arm','base_arm','variant']};rr=[x for x in f['summaries'] if all(x[k]==e[k] for k in ['kind','seed','arm'])]
  for region in ['changed_input_positions','final_position']:
   site='source' if region=='changed_input_positions' else 'final';gg=[x for x in f['geometry'] if x['region']==region and all(x[k]==e[k] for k in ['kind','seed','arm'])];out[site+'_local']=sum(x['delta_error'] for x in gg)/sum(x['target_delta_energy'] for x in gg)
   for corner in ['reconstructed_native','native_reconstructed','reconstructed_reconstructed']:
    ss=[x for x in rr if x['region']==region and x['corner']==corner];pool=next(x for x in ss if x['family'] is None);families=[x for x in ss if x['family'] is not None];pre=site+'_'+corner;out[pre+'_family']=sum(x['relative_response_error'] for x in families)/len(families);out[pre+'_pooled']=pool['relative_response_error'];out[pre+'_margin']=pool['relative_margin_error']
  ns=next(x for x in n['summaries'] if all(x[k]==e[k] for k in ['kind','seed','arm']))
  for k in ['mean_KL','delta_CE','top1_agreement','local_relative_error','active_mean']:out['natural_'+k]=ns[k]
  ir=[x for x in intacts if all(x[k]==e[k] for k in ['kind','seed','arm'])]
  for k in ['KL','binary_correct','task_top1','selected_mean','nonzero_mean']:out['intact_'+k]=sum(x[k] for x in ir)/len(ir)
  rows.append(out)
 comparisons=[];base=[x for x in models if x['variant']=='original'];metrics=[k for k in rows[0] if k not in ['kind','seed','arm','base_arm','variant']]
 for lhs,rhs in [('own_support_LS','original'),('full_dictionary_LS','original'),('full_dictionary_LS','own_support_LS')]:
  for metric in metrics:
   higher=any(k in metric for k in ['top1_agreement','binary_correct','task_top1']);pairs=[]
   for e in base:
    left=next(x for x in rows if (x['kind'],x['seed'],x['base_arm'],x['variant'])==(e['kind'],e['seed'],e['base_arm'],lhs));right=next(x for x in rows if (x['kind'],x['seed'],x['base_arm'],x['variant'])==(e['kind'],e['seed'],e['base_arm'],rhs));pairs.append(dict(kind=e['kind'],seed=e['seed'],base_arm=e['base_arm'],left=left[metric],right=right[metric],left_better=bool(left[metric]>right[metric] if higher else left[metric]<right[metric])))
   comparisons.append(dict(left_variant=lhs,right_variant=rhs,metric=metric,left_better=sum(x['left_better'] for x in pairs),pairs=pairs))
 op=V/'least_squares_evaluation_comparison.json';assert not op.exists();op.write_text(json.dumps(dict(status='completed',source_sha256=sha(Path(__file__)),factorial_sha256=sha(fp),natural_sha256=sha(np_),rows=rows,comparisons=comparisons,intact_rows=intacts,scope='All54 encoders; same18 dictionaries. Dense128 vs<=32 active coordinates is not a matched-cost architecture comparison. Familiar evaluation pairs, no new labels or training.'),indent=2)+'\n');print([(x['left_variant'],x['right_variant'],x['metric'],x['left_better']) for x in comparisons if x['metric'].endswith('_family') or x['metric']=='natural_mean_KL'])
if __name__=='__main__':main()

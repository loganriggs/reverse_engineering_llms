"""M97 inventory, replay-metric, independent tangent and derivative-accounting audit.

Derivative values themselves use the preflight-checked native autodiff runtime;
this is not an independent transformer Jacobian implementation.
"""
import json,sys,math
import numpy as np
import torch
from fit_pca_encoder import V,A,sha

def lp(t):
 x=t.detach().cpu().double().numpy();x-=x.max();return x-np.log(np.exp(x).sum())

def main():
 cohort=sys.argv[1];assert cohort=='source';torch.set_num_threads(1);rp=V/f'product_dictionary_direction_{cohort}_v2.json';r=json.loads(rp.read_text());pp=V/'product_dictionary_direction_execution_v2.json';assert r['protocol_sha256']==sha(pp) and r['source_sha256']==sha(V.parent/'score_product_dictionary_direction_v2.py');protocol=json.loads(pp.read_text())
 for name,digest in protocol['references'].items():assert sha(V/name)==digest
 for name,digest in protocol['sources'].items():assert sha(V.parent/name)==digest
 tp=json.loads((V/'balanced_response_training_protocol.json').read_text());den={(x['family'],x['region']):x['denominator'] for x in tp['scales']};models=json.loads((V/'product_dictionary_training_results.json').read_text())['models'];directions={}
 for seed in [71,72]:
  states=[]
  for arm in ['intact_KL','joint_three_corner']:
   entry=next(x for x in models if x['seed']==seed and x['arm']==arm);assert sha(entry['path'])==entry['sha256'];states.append(torch.load(entry['path'],weights_only=True)['state'])
  assert all(torch.equal(states[0][k],states[1][k]) for k in states[0] if k!='coefficients');s={k:v.double().numpy() for k,v in states[0].items()};dc=states[1]['coefficients'].double().numpy()-s['coefficients'];directions[seed]=(s,dc)
 prior_path=V/('product_dictionary_training_targets.json' if cohort=='training' else 'product_dictionary_source_factorial.json');assert r['prior_result_sha256']==sha(prior_path);prior=json.loads(prior_path.read_text());reference={};inputs={};native_writes={}
 if cohort=='training':
  inv=json.loads((V/'input_training_inventory.json').read_text());assert sha(inv['path'])==inv['sha256'];cache=torch.load(inv['path'],weights_only=True)['inputs'];inputs={k:x['input'] for k,x in cache.items()};oldreceipt=json.loads((V/'encoded_change_training_receipt.json').read_text());path=A/'emlp_m60_training_cache.pt';assert sha(path)==oldreceipt['cache_sha256'];native_writes={k:x['write'] for k,x in torch.load(path,weights_only=True)['inputs'].items()}
  for file in prior['files']:
   assert sha(file['path'])==file['sha256'];samples=torch.load(file['path'],weights_only=True)
   for sample in samples:
    row=prior['rows'][sample['row']];reference[row['seed'],row['arm'],row['assignment'],row['key']]=(row['K'],row['B'],{(b['region'],b['corner']):b['R'] for b in sample['branches']})
  assignments={seed:tp['schedules'][str(seed)]['task_updates'] for seed in [71,72]}
 else:
  native=json.loads((V/'source_holdout_native.json').read_text());assert sha(native['samples_path'])==native['samples_sha256'];ns=torch.load(native['samples_path'],weights_only=True);pairs=json.loads((V/'source_holdout_native_protocol.json').read_text())['pairs'];assignments={seed:[dict(pair_id=p['id'],family=p['family']) for p in pairs] for seed in [71,72]};raw={(x['seed'],x['arm'],x['key'],x['region'],x['corner']):x['response_error'] for x in prior['rows']}
  for file in prior['files']:
   assert sha(file['path'])==file['sha256'];blob=torch.load(file['path'],weights_only=True,mmap=True);pid=file['pair_id'];ai=next(i for i,p in enumerate(pairs) if p['id']==pid);family=pairs[ai]['family']
   for ep in [0,1]:
    key=f'{pid}|{ep}';inputs[key]=blob['inputs'][ep];native_writes[key]=blob['writes'][ep];n=lp(ns['baselines'][key]['logits'])
    for seed in [71,72]:
     for arm in ['intact_KL','joint_three_corner']:
      base=blob['baselines'][f'product_dictionary:{seed}:{arm}|{ep}'];K=float((np.exp(n)*(n-lp(base))).sum());br={(region,corner):raw[seed,arm,key,region,corner] for region in ['changed_input_positions','final_position','identity'] for corner in ['native_reconstructed','reconstructed_native','reconstructed_reconstructed']};B=sum(value/den[family,region] for (region,corner),value in br.items() if region!='identity')/6;reference[seed,arm,ai,key]=(K,B,br)
 maximum=0.;max_tangent_energy=0.;seen=set();checks=0;identity_checks=0;tangent_cache={}
 for row in r['rows']:
  ident=row['seed'],row['arm'],row['assignment'],row['key'];assert ident not in seen;seen.add(ident);K,B,br=reference[ident];assignment=assignments[row['seed']][row['assignment']];assert row['key'].rsplit('|',1)[0]==assignment['pair_id'] and row['family']==assignment['family'];assert row['anchor']==['intact_KL','joint_three_corner'].index(row['arm'])
  def check(a,b):
   nonlocal maximum
   assert np.isclose(a,b,rtol=1e-7,atol=1e-9),(a,b);maximum=max(maximum,abs(a-b))
  check(row['K'],K);check(row['B'],B);assert math.isfinite(row['K_slope']) and math.isfinite(row['B_slope']);assert {(b['region'],b['corner']) for b in row['branches']}==set(br);terms=[]
  for b in row['branches']:
   check(b['R'],br[b['region'],b['corner']]);assert math.isfinite(b['slope']);checks+=1
   if b['region']=='identity':assert b['R']==b['slope']==0;identity_checks+=1
   else:terms.append(b['slope']/den[row['family'],b['region']])
  assert abs(row['B_slope']-sum(terms)/len(terms))<=1e-12;discrepancy=abs(row['B_slope_direct']-row['B_slope']);assert discrepancy==row['B_slope_accounting_discrepancy'];assert row['B_slope_accounting_passed']==(discrepancy<=1e-7+1e-5*abs(row['B_slope_direct']))
  key=row['seed'],row['key']
  if key not in tangent_cache:
   s,dc=directions[row['seed']];z=inputs[row['key']].double().numpy();h=((z@s['left'].T)*(z@s['right'].T))/s['rms'];tangent=(h@dc.T)@s['dictionary'].T;tangent_cache[key]=float(np.square(tangent).sum())
  want=tangent_cache[key];relative=abs(row['write_direction_energy']-want)/max(want,1e-30);assert relative<=1e-5;max_tangent_energy=max(max_tangent_energy,relative);check(row['native_write_energy'],float(native_writes[row['key']].double().square().sum()));assert math.isfinite(row['secant_rounding_error_energy']) and row['secant_rounding_error_energy']>=0
 assert seen==set(reference);assert len(seen)==(528 if cohort=='training' else 112);assert checks==r['replayed_response_vectors']==(4740 if cohort=='training' else 1008);assert r['forward_count']==(3684 if cohort=='training' else 784)
 result=dict(status='passed',source_sha256=sha(__file__),result_sha256=sha(rp),rows=len(seen),response_values=checks,identity_values=identity_checks,direct_B_accounting_flags=[{k:x[k] for k in ['seed','key','anchor','B_slope','B_slope_direct','B_slope_accounting_discrepancy']} for x in r['rows'] if not x['B_slope_accounting_passed']],max_direct_B_accounting_discrepancy=max(x['B_slope_accounting_discrepancy'] for x in r['rows']),max_metric_absolute_error=maximum,max_tangent_energy_relative_error=max_tangent_energy,scope='Exact model/schedule/case inventory and allR/K/B values compared with prior independently audited fullvectors; NumPy affine tangent energies independent ofTorch producer, canonical weighted branch slope definition checked; original direct-B numerical consistency flags retained, not converted into passes. Native derivative accuracy is supported by fixed training preflight, not a second transformer Jacobian.')
 out=V/f'product_dictionary_direction_{cohort}_v2_audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()

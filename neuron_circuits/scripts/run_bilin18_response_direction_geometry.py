"""H826 saved full-vocabulary response direction analysis, no model forwards."""
import json,time,itertools
from pathlib import Path
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_early_local_geometry import sha
S='bilin18_response_direction_geometry'
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30))
def erank(g):return float(g.trace().square()/g.square().sum())
def corr(x,y):
 return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-12 and np.std(y)>1e-12 else None
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'captures':ART/'bilin18_fresh_geometry_causal_captures.pt','causal':ART/'bilin18_fresh_geometry_causal.json','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H826','hashes':hashes,'design':'All56 frozen receivers; six common-norm directions (near/far x block/PCA/random). Response is edited minus SAME RECEIVER zero-edit logits, weighted-centered under native recipient probabilities then multiplied by sqrt(p). Uniform-centered logit response is secondary.','metric':'Cosine of Fisher-local logit response vectors; finite changes so not an exact KL metric. Gram participation rank of six normalized directions measures angular concentration, not full Jacobian rank.','noise':'Weighted norm of zero-edit minus native logits; mark pairs as >10x replay discrepancy for both responses; retain all rows and no outcome-based removal.','scope':'Descriptive posthoc analysis on H825; no out-of-sample prediction, semantic labels or new forwards.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');d=torch.load(paths['captures'],weights_only=False);records=[];receivers=[];maxnormerr=0
 keys=[(arm,mode) for mode in ['block','pca','random'] for arm in ['near','far']]
 for i,pair in enumerate(d['pairs']):
  native=d['native'][pair['receiver']].double();prob=native.softmax(0);zero=d['outputs'][i,'zero','zero'].double();noise=zero-native;noise=(noise-(prob*noise).sum())*prob.sqrt();noise=float(noise.norm());patch=[];fisher=[];uniform=[];snr=[]
  for arm,mode in keys:
   v=d['patches'][i,arm,mode].double();delta=d['outputs'][i,arm,mode].double()-zero;w=(delta-(prob*delta).sum())*prob.sqrt();u=delta-delta.mean();patch.append(v/v.norm());fisher.append(w/w.norm().clamp_min(1e-30));uniform.append(u/u.norm().clamp_min(1e-30));snr.append(float(w.norm())/max(noise,1e-30));maxnormerr=max(maxnormerr,abs(float(v.norm())/pair['common_norm']-1))
  X=torch.stack(patch);Y=torch.stack(fisher);U=torch.stack(uniform);gx=X@X.T;gy=Y@Y.T;gu=U@U.T
  for a,b in itertools.combinations(range(6),2):
   records.append({'pair':i,'site':pair['site'],'block':pair['block'],'receiver':pair['receiver'],'a':keys[a],'b':keys[b],'input_cosine':float(gx[a,b]),'output_fisher_cosine':float(gy[a,b]),'output_uniform_cosine':float(gu[a,b]),'above_noise':min(snr[a],snr[b])>10,'min_signal_noise_ratio':min(snr[a],snr[b]),'same_method':keys[a][1]==keys[b][1]})
  rr=records[-15:];receivers.append({'pair':i,'site':pair['site'],'block':pair['block'],'input_direction_rank':erank(gx),'output_fisher_direction_rank':erank(gy),'output_uniform_direction_rank':erank(gu),'angle_correlation':corr([r['input_cosine'] for r in rr],[r['output_fisher_cosine'] for r in rr]),'mean_absolute_cosine_distortion':float((gx-gy).abs().triu(1).sum()/15),'min_signal_noise_ratio':min(snr)})
 summary=[]
 for site in ['residual0','residual1']:
  rec=[r for r in receivers if r['site']==site];summary.append({'site':site,'kind':'six_direction_geometry','receivers':len(rec),'median_input_rank':float(np.median([r['input_direction_rank'] for r in rec])),'median_fisher_rank':float(np.median([r['output_fisher_direction_rank'] for r in rec])),'median_uniform_rank':float(np.median([r['output_uniform_direction_rank'] for r in rec])),'median_within_receiver_angle_correlation':float(np.median([r['angle_correlation'] for r in rec])),'median_absolute_cosine_distortion':float(np.median([r['mean_absolute_cosine_distortion'] for r in rec]))})
  for mode in ['block','pca','random']:
   rr=[r for r in records if r['site']==site and r['same_method'] and r['a'][1]==mode];summary.append({'site':site,'kind':mode+'_near_far','median_input_cosine':float(np.median([r['input_cosine'] for r in rr])),'median_fisher_cosine':float(np.median([r['output_fisher_cosine'] for r in rr])),'median_uniform_cosine':float(np.median([r['output_uniform_cosine'] for r in rr])),'above_noise':sum(r['above_noise'] for r in rr),'n':len(rr)})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'common_norm_relative_error':maxnormerr,'finite':all(np.isfinite(r['output_fisher_cosine']) for r in records),'pairwise_rows':len(records),'all_rows_above_noise':all(r['above_noise'] for r in records),'minimum_signal_noise_ratio':min(r['min_signal_noise_ratio'] for r in records),'whole_model_forwards':0,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'];(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'receivers':receivers,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits},indent=2))
if __name__=='__main__':main()

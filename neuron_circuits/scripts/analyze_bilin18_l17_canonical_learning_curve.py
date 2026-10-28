import json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_canonical_learning_curve'
def main():
 j=json.loads(STEM.with_suffix('.json').read_text());z=np.load(STEM.with_name(STEM.name+'_rows.npz'));cfg=j['config'];rng=np.random.default_rng(cfg['bootstrap_seed']);ix=(rng.integers(0,8,(cfg['bootstrap_samples'],64,8))*64+np.arange(64)[None,:,None]).reshape(cfg['bootstrap_samples'],512);est=[];draw=[];names=[];concentration={}
 for sign,sl in [('negative',slice(0,512)),('positive',slice(512,1024))]:
  effect=z['confirmation_effects'][sl,0];large=z['confirmation_n1024'][sl,0];concentration[sign]={'effect_effective_count':float(effect.sum()**2/(effect**2).sum()),'largest_effect_fraction':float(effect.max()/effect.sum()),'largest26_effect_fraction':float(np.sort(effect)[-26:].sum()/effect.sum())}
  for reference in ['original','n256']:
   delta=z['confirmation_'+reference][sl,0]-large;names.append('n1024_minus_'+reference+'_'+sign);est.append(delta.sum()/effect.sum());draw.append(delta[ix].sum(1)/effect[ix].sum(1))
 est=np.array(est);draw=np.stack(draw,-1);sd=draw.std(0,ddof=1);assert np.all(sd>0);q=np.quantile(np.max(np.abs((draw-est)/sd),axis=1),.95);lo=est-q*sd;hi=est+q*sd;j['primary_statistics']={name:{'estimate':float(e),'joint95_interval':[float(l),float(h)],'pass_lower_gt_zero':bool(l>0)} for name,e,l,h in zip(names,est,lo,hi)};j['bootstrap']={'samples':len(ix),'simultaneous_critical_value':float(q),'method':'Paired resampling of eight receiving contexts within each fixed64 donor stratum; same indices for both signs and models. Joint4 standardized bootstrap bands; conditional on fixed fitted models, donor pool and calibration.'};j['confirmation_effect_concentration']=concentration;STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print(j['primary_statistics']);print(concentration)
if __name__=='__main__':main()

"""H291 residual-response covariance directions, nested ridge readouts."""
from pathlib import Path
import json
import numpy as np
from bilin18_conditional_response_basis import conditional_statistics,directions
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_response_conditional'
def main():
 cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());d=np.load(ART/'bilin18_l17_response_informed_statistics.npz');keys=[k[:-5] for k in d.files if k.endswith('_gram')];pr=np.load(ART/'bilin18_l17_residual_cross_actual_refined_program.npz');f=np.load(ART/'bilin18_l17_fixed_readout_frozen.npz');outside=np.load(ART/'bilin18_l17_feature_outside_frozen.npz');ridge=np.load(ART/'bilin18_l17_response_ridge_frozen.npz');reads=np.concatenate([pr['core'],pr['partner']],1).astype('float64');q=np.linalg.qr(reads)[0];a=reads/f['feature_std'];g=sum(d[k+'_gram'] for k in keys if k.startswith('train'));r=sum(d[k+'_rhs'] for k in keys if k.startswith('train'));complete=np.linalg.qr(reads,mode='complete')[0];complement=complete[:,64:];conditional,cross=conditional_statistics(g,r,a,complement);maps={};basis_audits={}
 for beta in cfg['betas']:
  basis,audit=directions(conditional,cross,complement,beta,64);basis_audits[str(beta)]=audit;scale=np.sqrt(np.einsum('ik,ij,jk->k',basis,d['state_cov'],basis)).clip(1e-6);basis=basis/scale
  for rank in [32,64]:maps[f'b{beta}_rank{rank}']=np.concatenate([a,basis[:,:rank]],1)
 records={};weights={};projected={}
 for family,m in maps.items():
  pg=m.T@g@m;rr=m.T@r;width=m.shape[1];groupstats={k:(m.T@d[k+'_gram']@m,m.T@d[k+'_rhs'],float(d[k+'_energy'])) for k in keys}
  for alpha in cfg['alphas']:
   h=pg+alpha*np.trace(pg)/width*np.eye(width);w=np.linalg.solve(h,rr);name=family+'_a'+str(alpha);weights[name]=w;v={'family':family,'alpha':alpha,'normal_equation_relative_residual':float(np.linalg.norm(h@w-rr)/np.linalg.norm(rr)),'groups':{}}
   for k,(gg,r1,e) in groupstats.items():v['groups'][k]=float((np.sum(w*(gg@w))-2*np.sum(w*r1)+e)/e)
   for split in ['train','validation']:v[split]=sum(v['groups'][k]*float(d[k+'_energy']) for k in keys if k.startswith(split))/sum(float(d[k+'_energy']) for k in keys if k.startswith(split))
   records[name]=v
  print('family',family,'complete',flush=True)
 winners={family:min((k for k,v in records.items() if v['family']==family),key=lambda k:records[k]['validation']) for family in maps};bestinf=min(winners.values(),key=lambda k:records[k]['validation']);old=json.loads((ART/'bilin18_l17_response_informed.json').read_text());bestnat=old['best_natural'];gain=1-records[bestinf]['validation']/old['records'][bestnat]['validation'];basegain=1-records[bestinf]['validation']/old['records'][old['winners']['base64']]['validation'];audits=basis_audits;np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),state_mean=d['state_mean'],**{'map_'+k:v for k,v in maps.items()},**weights);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'records':records,'winners':winners,'best_informed':bestinf,'best_natural':bestnat,'direction_gain':gain,'base64_gain':basegain,'audits':audits,'gates':{'direction_gain':gain>=.02,'base64_gain':basegain>=.02}},indent=2)+'\n');print('WINNERS',winners,'GAIN',gain,basegain,flush=True)
if __name__=='__main__':main()

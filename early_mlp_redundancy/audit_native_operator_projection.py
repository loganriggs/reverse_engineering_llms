"""NumPy polynomial replay of every native operator error-decomposition case."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha
from native_projected_quadratic import write,action,prediction

def main():
 torch.set_num_threads(1);rp=V/'native_operator_projection_decomposition.json';r=json.loads(rp.read_text());assert r['source_sha256']==sha(V.parent/'decompose_native_operator_projection.py');assert sha(r['samples_path'])==r['samples_sha256'];samples=torch.load(r['samples_path'],weights_only=True);ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];z=torch.load(ref['path'],weights_only=True);L,R,D=[z[n].double().numpy() for n in ['L','R','D']];mean=z['mean'].numpy();fit=json.loads((V/'native_operator_metric_fits.json').read_text());assert sha(fit['path'])==fit['sha256'];models=torch.load(fit['path'],weights_only=True)['models'];seen=set();maximum=0.;maxvector=0.
 for sample in samples:
  i=sample['row'];assert i not in seen;seen.add(i);row=r['rows'][i];m=models[row['basis']];k=row['rank'];E=m['encoder'][:,:k].numpy();B=m['decoder'][:,:k].numpy();zr=sample['recipient'].numpy();zd=sample['donor'].numpy();delta=zd-zr;h=(zr+zd)/2-mean;ph=(h@E)@B.T;pd=(delta@E)@B.T
  exact=write(L,R,D,zd)-write(L,R,D,zr);anchor=action(L,R,D,mean+ph,delta);consistent=prediction(L,R,D,mean,E,B,zd)-prediction(L,R,D,mean,E,B,zr);a=action(L,R,D,h-ph,delta);b=action(L,R,D,ph,delta-pd)
  for name,want in [('exact',exact),('anchor',anchor),('consistent',consistent),('context_term',a),('direction_term',b)]:
   got=sample[name].numpy();err=float(np.linalg.norm(want-got));assert err<=1e-8*np.linalg.norm(want)+1e-8,(i,name,err);maxvector=max(maxvector,err)
  def check(name,want):
   nonlocal maximum
   err=abs(want-row[name]);assert np.isclose(want,row[name],rtol=1e-8,atol=1e-8),(i,name,want,row[name]);maximum=max(maximum,err)
  check('target_energy',float(np.square(exact).sum()));check('anchor_error',float(np.square(exact-anchor).sum()));check('consistent_error',float(np.square(exact-consistent).sum()));check('context_term_energy',float(np.square(a).sum()));check('direction_term_energy',float(np.square(b).sum()));check('twice_cross',float(2*(a*b).sum()));assert np.allclose(exact-consistent,a+b,rtol=1e-8,atol=1e-8)
 assert seen==set(range(2448));ident=lambda x:(x['cohort'],x['pair_id'],x['endpoint'],x['region'],x['basis'],x['rank']);assert len({ident(x) for x in r['rows']})==2448
 summaries=[]
 for cohort in ['whole','source']:
  for region in ['changed_input_positions','final_position']:
   for basis in models:
    for rank in [4,16,64]:
     rr=[x for x in r['rows'] if (x['cohort'],x['region'],x['basis'],x['rank'])==(cohort,region,basis,rank)];den=sum(x['target_energy'] for x in rr);s=dict(cohort=cohort,region=region,basis=basis,rank=rank,n=len(rr),target_energy=den)
     for field in ['anchor_error','consistent_error','context_term_energy','direction_term_energy','twice_cross']:s[field+'_relative']=sum(x[field] for x in rr)/den
     summaries.append(s)
 out=dict(status='passed',source_sha256=sha(__file__),result_sha256=sha(rp),cases=2448,max_vector_absolute_error=maxvector,max_metric_absolute_error=maximum,summaries=summaries,scope='Everycase independentlyreplayedwithNumPy polynomial writes/finitechanges/contextactions; independentlocalerror metrics anddecomposition. Pooledlocalratios, notdownstreamresponses ornativechannelcancellation.')
 f=V/'native_operator_projection_decomposition_audit.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print('PASSED',len(seen),maxvector,maximum)
if __name__=='__main__':main()

"""M67 independent applied-vector reconstruction and final-response audit."""
from pathlib import Path
import json,torch,numpy as np
from audit_support_change_diagnostic import sha,lp,margin
P=Path(__file__).resolve().parent;V=P/'validation'
def main():
 torch.set_num_threads(1);rp=V/'native_change_subset_responses.json';r=json.loads(rp.read_text());atlas=json.loads((V/'native_change_coordination.json').read_text());assert r['status']=='completed' and r['source_sha256']==sha(P/'probe_native_change_subsets.py') and r['atlas_sha256']==sha(V/'native_change_coordination.json');wp=Path(atlas['weights_path']);assert sha(wp)==atlas['weights_sha256'];weights=torch.load(wp,weights_only=True);mp=Path(atlas['masks_path']);assert sha(mp)==atlas['masks_sha256'];bank=torch.load(mp,weights_only=True).numpy();masks=np.stack([bank[0],1-bank[0],bank[1],1-bank[1]]);pairs={}
 for pn in ['training_causal_coverage_protocol.json','source_holdout_native_protocol.json']:pairs.update({x['id']:x for x in json.loads((V/pn).read_text())['pairs']})
 n=0;nt=0;maxmetric=0.;maxdelta=0.
 for fi,file in enumerate(r['files']):
  path=Path(file['path']);assert sha(path)==file['sha256'];b=torch.load(path,weights_only=True);af=next(x for x in atlas['files'] if x['pair_id']==file['pair_id']);ap=Path(af['path']);assert sha(ap)==af['sha256'];orig=torch.load(ap,weights_only=True);pair=pairs[file['pair_id']]
  for tr in [x for x in r['teachers'] if x['pair_id']==file['pair_id']]:
   ep=tr['endpoint'];l=tr['layer'];reg=tr['region'];sid=f'{ep}|{l}|{reg}';spec=next(x for x in atlas['rows'] if x['pair_id']==file['pair_id'] and x['endpoint']==ep and x['layer']==l and x['region']==reg);pos=spec['recipient_positions'];dp=spec['donor_positions'];cr=orig['captures'][ep][l];cd=orig['captures'][1-ep][l];dc=(cd['products'][:,dp].numpy()-cr['products'][:,pos].numpy()).astype(float);D=weights[l]['Down'].double().numpy();native=(cd['write'][:,dp].numpy()-cr['write'][:,pos].numpy()).astype(float);assert np.array_equal(b['full_deltas'][sid].numpy(),native);base=lp(orig['logits'][ep]);tlp=lp(b['teachers'][sid]);target=tlp-base;g=pair['answer_groups'][ep];v=pair['variants'][ep];tm=margin(tlp,g,v)-margin(base,g,v);te=float((np.exp(base)*target**2).sum());ne=float((native**2).sum());assert np.isclose(te,tr['target_energy'],rtol=2e-9,atol=2e-10);assert np.isclose(tm,tr['margin_change'],rtol=2e-9,atol=2e-10);assert tr['native_binary']==(margin(base,g,v)>0);assert tr['native_top1']==(int(orig['logits'][ep].argmax()) in g[v]);nt+=1
   for row in [x for x in r['rows'] if x['status']=='evaluated' and all(x[k]==tr[k] for k in ['pair_id','endpoint','layer','region'])]:
    key=sid+f"|{row['mask']}|{row['factor']}";delta=b['deltas'][key].double().numpy();recon=(dc*masks[row['mask']])@D.T*row['factor'];rel=float(((delta-recon)**2).sum())/max(1.,float((recon**2).sum()));maxdelta=max(maxdelta,rel);assert rel<1e-10;slp=lp(b['changed'][key]);er=float((np.exp(base)*(slp-base-target)**2).sum());me=(margin(slp,g,v)-margin(base,g,v)-tm)**2
    for k,val in [('response_error',er),('target_energy',te),('margin_error',me),('margin_target_energy',tm**2)]:maxmetric=max(maxmetric,abs(val-row[k]));assert np.isclose(row[k],val,rtol=2e-9,atol=2e-10)
    for k,val in [('local_error',float(((delta-native)**2).sum())),('native_delta_energy',ne),('delta_energy',float((delta**2).sum()))]:assert np.isclose(row[k],val,rtol=1e-12,atol=1e-7)
    if reg=='identity':assert er==te==ne==0
    n+=1
  if (fi+1)%10==0:print('AUDIT_PAIRS',fi+1,flush=True)
 assert n==12048 and nt==1506 and sum(x['status']=='ineligible' for x in r['rows'])==6
 op=V/'native_change_subset_audit.json';assert not op.exists();op.write_text(json.dumps(dict(status='passed',source_sha256=sha(Path(__file__)),helper_sha256=sha(P/'audit_support_change_diagnostic.py'),result_sha256=sha(rp),response_vectors=n,teacher_vectors=nt,ineligible=6,max_applied_delta_relative_squared_error=maxdelta,max_response_metric_abs_error=maxmetric,scope='All masked directions reconstructed independently in NumPy from saved products/masks/weights; every local error and full-vocabulary final response checked. Float64 reconstruction checks float32 applied vectors to reported tolerance. Producer exactly replays all1338 old teachers.'),indent=2)+'\n');print('PASS',n,nt,maxdelta,maxmetric,flush=True)
if __name__=='__main__':main()

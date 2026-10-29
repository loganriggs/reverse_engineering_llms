"""M71 verify that fixed full-span changes equal independent affine LS encoding."""
from pathlib import Path
import json,torch,numpy as np
from torch.nn import functional as F
from prepare_encoded_span_diagnostic import sha
P=Path(__file__).resolve().parent;V=P/'validation'
def main():
 torch.set_num_threads(1);geo=json.loads((V/'encoded_span_geometry.json').read_text());old=json.loads((V/'encoded_factorial.json').read_text());models=json.loads((V/'encoded_change_evaluation_models.json').read_text())['models'];states={};n=0;nd=0;maxdelta=0.;maxlocal_excess=0.
 for e in models:
  p=Path(e['path']);assert sha(p)==e['sha256'];s=torch.load(p,weights_only=True);D=F.normalize(s['state']['dictionary'],dim=0).double().numpy();u,sv,vt=np.linalg.svd(D,full_matrices=False);keep=sv>1e-10*sv[0];pinv=(vt[keep].T/sv[keep])@u[:,keep].T;states[tuple(e[k] for k in ['kind','seed','arm'])]=(s,D,pinv)
 for file in old['files']:
  p=Path(file['path']);assert sha(p)==file['sha256'];b=torch.load(p,weights_only=True);gf=next(x for x in geo['files'] if x['pair_id']==file['pair_id']);gp=Path(gf['path']);assert sha(gp)==gf['sha256'];gb=torch.load(gp,weights_only=True)
  for e in models:
   mk=tuple(e[k] for k in ['kind','seed','arm']);ms=':'.join(map(str,mk));s,D,pinv=states[mk];mu=s['mean'].double().numpy();scale=float(s['scale']);bias=s['state']['bias'].double().numpy();recs=[]
   for ep in [0,1]:
    w=b['writes'][ep].double().numpy();x=(w-mu)/scale;code=(x-bias)@pinv.T;rec=mu+scale*(code@D.T+bias);actual=b['reconstructions'][ms+f'|{ep}']['write'].double().numpy();excess=(np.square(rec-w).sum()-np.square(actual-w).sum())/max(np.square(w).sum(),1.);maxlocal_excess=max(maxlocal_excess,float(excess));assert excess<1e-6;recs.append(rec);n+=1
   for entry in [x for x in gb['entries'] if all(x['identity'][k]==e[k] for k in ['kind','seed','arm'])]:
    ep=int(entry['identity']['key'].rsplit('|',1)[1]);pos=entry['positions'];dp=entry['donor_positions'];diff=recs[1-ep][:,dp]-recs[ep][:,pos];projection=entry['deltas']['dictionary'].double().numpy();rel=np.square(diff-projection).sum()/max(np.square(projection).sum(),1.);maxdelta=max(maxdelta,float(rel));assert rel<1e-8;nd+=1
 assert n==504 and nd==1512
 op=V/'dense_span_encoder_equivalence.json';assert not op.exists();op.write_text(json.dumps(dict(status='passed',source_sha256=sha(Path(__file__)),geometry_sha256=sha(V/'encoded_span_geometry.json'),models_sha256=sha(V/'encoded_change_evaluation_models.json'),independent_reconstructions=n,difference_checks=nd,max_relative_squared_difference=maxdelta,max_local_error_excess=maxlocal_excess,formula='a(w)=D^+[(w-mu)/scale-b]; f(w)=mu+scale*(D a(w)+b). Therefore f(w_d)-f(w_r)=P_D(w_d-w_r).',scope='Every endpoint encoded independently without pair labels/targets. Float64 affine least squares; numerical differences versus M71 reflect native-delta/applied-vector float32 rounding. No downstream replay of this independent-encoding implementation yet. Dense128 active coordinates versus original<=32 is a cost change, not matched-sparsity superiority.'),indent=2)+'\n');print('PASS',n,nd,maxdelta)
if __name__=='__main__':main()

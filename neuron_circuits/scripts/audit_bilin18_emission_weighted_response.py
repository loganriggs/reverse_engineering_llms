import json,hashlib
from pathlib import Path
import torch,numpy as np
from run_bilin18_diverse_causal_tasks import ART
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cp=ART/'bilin18_emission_logit_response_captures.pt';mp=ART/'bilin18_emission_surface_transfer_manifest.json';d=torch.load(cp,weights_only=False);rows=json.loads(mp.read_text());G=d['tangent'].double();Z=d['logits'];P=Z[0.].double().softmax(-1);fits={};audits={}
 for name,ii in [('capital',[i for i,e in enumerate(rows) if e['item']<4 and e['family']=='capital' and e['form']==0]),('pooled',[i for i,e in enumerate(rows) if e['item']<4])]:
  p=P[ii];g=G[ii];D=p.sum(0);b0=(p*g).sum(0)/D;M=torch.eye(len(ii),dtype=torch.float64)-(p/D)@p.T;rhs=(p*g).sum(-1)-p@b0;c=torch.linalg.pinv(M,hermitian=True,rtol=1e-10)@rhs;b=b0-(p.T@c)/D;b-=b.mean();fits[name]=b
  residual=g-b;centered=residual-(residual*p).sum(-1,keepdim=True);grad=(p*centered).sum(0);audits[name]=dict(normal_equation_max_abs=float(grad.abs().max()),train_weighted_mse=float((p*centered.square()).sum()/len(ii)),fit_ids=[rows[i]['id'] for i in ii])
 records=[]
 for i,e in enumerate(rows):
  if e['item']<4:continue
  p=P[i];g=G[i];z=Z[0.][i].double()
  def center(x):return x-(p*x).sum()
  def norm(x):return (center(x).square()*p).sum().sqrt()
  for name,b in fits.items():
   gain=float((center(g)*center(b)*p).sum()/(center(b).square()*p).sum());actual=Z[.05][i].double().log_softmax(0);pred=(z+.05*b).log_softmax(0);base=z.log_softmax(0);records.append(dict(family=e['family'],form=e['form'],id=e['id'],fit=name,tangent_error=float(norm(g-b)/norm(g)),oracle_gain=gain,scalar_error=float(norm(g-gain*b)/norm(g)),prediction_kl=float((actual.exp()*(actual-pred)).sum()),omission_kl=float((actual.exp()*(actual-base)).sum())))
 summary={}
 for family in ['capital','language','plural']:
  for form in range(3):
   for name in fits:
    rr=[r for r in records if r['family']==family and r['form']==form and r['fit']==name];summary[f'{family}_{form}_{name}']=dict(tangent_error=float(np.mean([r['tangent_error'] for r in rr])),scalar_error=float(np.mean([r['scalar_error'] for r in rr])),relative_kl=float(np.mean([r['prediction_kl'] for r in rr])/np.mean([r['omission_kl'] for r in rr])))
 torch.save(fits,ART/'bilin18_emission_logit_response_weighted_fits.pt');out=dict(summary=summary,records=records,audits=audits,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),capture_sha256=hashlib.sha256(cp.read_bytes()).hexdigest());(ART/'bilin18_emission_logit_response_weighted_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2));print(audits)
if __name__=='__main__':main()

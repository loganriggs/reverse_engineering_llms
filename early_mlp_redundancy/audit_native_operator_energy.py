"""Independent direct uncentered energy check for the mean/variation decomposition."""
import json
import torch
from fit_pca_encoder import V,sha

def main():
 torch.set_num_threads(4);ep=V/'native_operator_metric_energy.json';e=json.loads(ep.read_text());f=json.loads((V/'native_operator_metric_fits.json').read_text());assert sha(f['path'])==f['sha256'];s=torch.load(f['path'],weights_only=True);r=json.loads((V/'native_input_reference.json').read_text());assert sha(r['path'])==r['sha256'];d=torch.load(r['path'],weights_only=True);X=d['z'][d['indices']].double();rows=[]
 for entry in e['rows']:
  Q=s['metrics'][entry['metric']];direct=float(((X@Q)*X).sum()/len(X));decomposed=entry['shared_mean_energy']+entry['centered_variation_energy'];rel=abs(direct-decomposed)/abs(direct);assert rel<1e-12;rows.append(dict(metric=entry['metric'],direct_uncentered_energy=direct,decomposed_energy=decomposed,relative_discrepancy=rel))
 out=dict(status='passed',source_sha256=sha(__file__),energy_sha256=sha(ep),rows=rows,scope='Directuncenteredquadraticenergy overall4092traininginputs independentlymatchesmean+centeredvariance. Taskdirectionmetric averagesdirections acrosscontexts; notempiricaljointcontext/directioncausalerror.');p=V/'native_operator_metric_energy_audit.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()

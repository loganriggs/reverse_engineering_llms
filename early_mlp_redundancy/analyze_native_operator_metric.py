"""Separate shared-mean and centered operator energy; no heldout claims."""
import json
import torch
from fit_pca_encoder import V,sha

def main():
 torch.set_num_threads(1);fp=V/'native_operator_metric_fits.json';f=json.loads(fp.read_text());ap=V/'native_operator_metric_audit.json';a=json.loads(ap.read_text());assert a['status']=='passed' and a['fit_sha256']==sha(fp);assert sha(f['path'])==f['sha256'];s=torch.load(f['path'],weights_only=True);rows=[]
 for r in f['rows']:
  name=r['metric'];mean=s['models'][name]['mean'];Q=s['metrics'][name];constant=float(mean@Q@mean);variation=r['training_operator_energy'];rows.append(dict(metric=name,shared_mean_energy=constant,centered_variation_energy=variation,shared_mean_fraction=constant/(constant+variation),centered_variation_fraction=variation/(constant+variation),ranks=[dict(rank=x['rank'],residual_fraction_of_centered_variation=x['training_metric_residual_ratio'],residual_fraction_of_total_energy=x['training_metric_residual_ratio']*variation/(constant+variation)) for x in r['capacity']]))
 out=dict(status='completed',source_sha256=sha(__file__),fit_sha256=sha(fp),audit_sha256=sha(ap),rows=rows,scope='Postfit descriptive energyidentity: E||J(z)||S²=||J(mean)||S²+E||J(z-mean)||S². Euclideanrow insteadmeasuresinputenergy. Trainingonly, notlocalfinitewriteerror/downstreamcausalfidelity; noselection.')
 p=V/'native_operator_metric_energy.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()

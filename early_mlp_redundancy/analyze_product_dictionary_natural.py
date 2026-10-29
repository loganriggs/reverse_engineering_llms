"""Within-seed natural-retention contrasts, all six frozen candidates retained."""
import json
import numpy as np
from fit_pca_encoder import V,sha

def main():
 rp=V/'product_dictionary_natural.json';r=json.loads(rp.read_text());ap=V/'product_dictionary_natural_audit.json';a=json.loads(ap.read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);rows=[]
 fields=['mean_KL','CE','delta_CE','top1_agreement','local_relative_error']
 for seed in [71,72]:
  byarm={arm:{x['document']:x for x in r['rows'] if x['seed']==seed and x['arm']==arm} for arm in ['initial','intact_KL','joint_three_corner']}
  for baseline,candidate in [('initial','intact_KL'),('initial','joint_three_corner'),('intact_KL','joint_three_corner')]:
   assert set(byarm[baseline])==set(byarm[candidate])==set(range(32,64));metrics={}
   for field in fields:
    diffs=[byarm[candidate][doc][field]-byarm[baseline][doc][field] for doc in range(32,64)];higher=field=='top1_agreement'
    metrics[field]=dict(mean_difference=float(np.mean(diffs)),improved_documents=sum(d>0 if higher else d<0 for d in diffs),equal_documents=sum(d==0 for d in diffs),document_differences=diffs)
   rows.append(dict(seed=seed,baseline=baseline,candidate=candidate,metrics=metrics))
 out=dict(status='completed',source_sha256=sha(__file__),result_sha256=sha(rp),audit_sha256=sha(ap),summaries=r['summaries'],comparisons=rows,scope='All32documents percomparison; descriptive paired contrasts, not independent trials or a percentage of computation understood.')
 dest=V/'product_dictionary_natural_comparison.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()

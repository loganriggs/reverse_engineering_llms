"""Fixed coverage-by-objective retention contrasts, no candidate selection."""
import json
from fit_pca_encoder import V,sha

def main():
 rp=V/'product_dictionary_coverage_natural.json';r=json.loads(rp.read_text());ap=V/'product_dictionary_coverage_natural_audit.json';a=json.loads(ap.read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);lookup={(s['seed'],s['arm']):s for s in r['summaries']};rows=[]
 for seed in [71,72]:
  for metric in ['mean_KL','CE','delta_CE','top1_agreement','local_relative_error']:
   vals={coverage:{objective:lookup[seed,coverage+'__'+objective][metric] for objective in ['intact_KL','joint_three_corner']} for coverage in ['full_pool','old_visited']}
   effects={c:x['joint_three_corner']-x['intact_KL'] for c,x in vals.items()}
   rows.append(dict(seed=seed,metric=metric,initial=lookup[seed,'initial'][metric],values=vals,joint_minus_KL=effects,interaction=effects['full_pool']-effects['old_visited'],joint_full_minus_control=vals['full_pool']['joint_three_corner']-vals['old_visited']['joint_three_corner']))
 out=dict(status='completed',source_sha256=sha(__file__),result_sha256=sha(rp),audit_sha256=sha(ap),rows=rows,scope='All32documents,2seeds,8finals+2originalinitials. Higher top1agreement isbetter; lower error/loss isbetter. Retention only, not causal fidelity ornewtask discovery.')
 f=V/'product_dictionary_coverage_natural_comparison.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()

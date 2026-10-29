"""Audited pooled movement summaries, preserving train/evaluation distinction."""
import json,math
from fit_pca_encoder import V,sha

def main():
 rp=V/'product_dictionary_movement.json';r=json.loads(rp.read_text());ap=V/'product_dictionary_movement_audit.json';a=json.loads(ap.read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);summaries=[]
 for parameter in r['coefficients']:
  for population in ['train','evaluation']:
   rows=[x for x in r['rows'] if x['seed']==parameter['seed'] and x['arm']==parameter['arm'] and x['population']==population];assert len(rows)==(1 if population=='train' else 32);total=lambda key:math.fsum(x[key] for x in rows);energy=total('native_energy')
   summaries.append(dict(**parameter,population=population,positions=sum(x['positions'] for x in rows),relative_write_movement=math.sqrt(total('correction_energy')/energy),relative_initial_output_movement=math.sqrt(total('correction_energy')/total('initial_output_energy')),relative_initial_error=total('initial_error')/energy,relative_final_error=total('final_error')/energy,relative_initial_projection_gap=math.sqrt(total('initial_projection_gap')/energy),relative_final_projection_gap=math.sqrt(total('final_projection_gap')/energy),relative_error_increase=(total('final_error')-total('initial_error'))/energy))
 out=dict(status='completed',source_sha256=sha(__file__),result_sha256=sha(rp),audit_sha256=sha(ap),summaries=summaries,scope='Ratios of pooled energies, then square root for movement norms; reconstruction error remains squared. Coefficient norm is parameterization-dependent; neither ratio is percent computation understood.')
 p=V/'product_dictionary_movement_comparison.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()

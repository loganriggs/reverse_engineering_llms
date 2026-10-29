"""M99 population Gram matrices and explicit infinitesimal Euclidean predictions."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha

def main():
 torch.set_num_threads(1);rp=V/'product_dictionary_component_gradients.json';r=json.loads(rp.read_text());ap=V/'product_dictionary_component_gradient_audit.json';a=json.loads(ap.read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);assert sha(a['samples_path'])==a['samples_sha256'];ind={(x['seed'],x['point']):x['gradients'] for x in torch.load(a['samples_path'],weights_only=True)};p=json.loads((V/'product_dictionary_component_execution.json').read_text());lam=p['joint_weight'];rows=[]
 for entry in r['models']:
  assert sha(entry['path'])==entry['sha256'];s=torch.load(entry['path'],weights_only=True);g={n:t.numpy().reshape(-1) for n,t in s['gradients'].items()};g['source']=sum(t for n,t in g.items() if n.startswith('source__'));g['final']=sum(t for n,t in g.items() if n.startswith('final__'));g['task_B']=g['source']+g['final'];g['KL_objective']=.5*(g['natural_K']+g['task_K']);g['joint_objective']=g['KL_objective']+.5*lam*g['task_B'];norms={n:float(np.linalg.norm(t)) for n,t in g.items()};gram=[]
  for n,u in g.items():
   for m,w in g.items():
    dot=float(u@w);den=norms[n]*norms[m];gram.append(dict(left=n,right=m,dot=dot,cosine=dot/den if den else None))
  alt={n:t.numpy().reshape(-1) for n,t in ind[entry['seed'],entry['point']].items()};alt['task_B']=alt['source']+alt['final'];alt['KL_objective']=.5*(alt['natural_K']+alt['task_K']);alt['joint_objective']=alt['KL_objective']+.5*lam*alt['task_B'];predictions=[]
  for loss in ['source','final','natural_K','task_K']:
   for direction in ['source','final','task_B','KL_objective','joint_objective']:
    slope=-float(g[loss]@g[direction]);other=-float(alt[loss]@alt[direction]);predictions.append(dict(loss=loss,negative_gradient_direction=direction,derivative_per_unit_step=slope,grouped_audit_derivative=other,sign_agrees=bool(np.sign(slope)==np.sign(other))))
  rows.append(dict(seed=entry['seed'],point=entry['point'],losses=s['losses'],norms=norms,gram=gram,predictions=predictions,near_zero_norms=[n for n,x in norms.items() if x<1e-12]))
 out=dict(status='completed',source_sha256=sha(__file__),result_sha256=sha(rp),audit_sha256=sha(ap),joint_weight=lam,rows=rows,scope='Infinitesimal unpreconditioned gradient directions in fixed coefficientcoordinates. Exact0.5natural/0.5taskweight. Source/final names mean theircontributions tobalancedB, includingendpoint-dependentdivisors. Rawstepderivatives arenotfiniteupdates orAdamtrajectories. Nearzeroflag threshold1e-12. Noheldoutgradients ornewfit.');f=V/'product_dictionary_component_gradient_comparison.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print('ANALYZED',len(rows))
if __name__=='__main__':main()

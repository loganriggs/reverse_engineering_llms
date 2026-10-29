"""M113 local encoder-coordinate rank on native MLP1 input-sphere tangents."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha


def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
 pp=V/'native_code_tangent_protocol.json';p=json.loads(pp.read_text());assert p['source_sha256']==sha(__file__)
 for name,h in p['bindings'].items():assert sha(V/name)==h
 wr=json.loads((V/'matched_layer_operator_inputs.json').read_text());assert sha(wr['path'])==wr['sha256'];w=torch.load(wr['path'],weights_only=True)['states'][1]
 L,R,D=[w[k].double().cuda() for k in ['L','R','D']];zs=w['z'].double().cuda();indices=p['indices'];models=json.loads((V/'continuous_training_results.json').read_text())['models'];rows=[];svdaudits=[];maxjvp=0.
 for entry in models:
  assert sha(entry['path'])==entry['sha256'];state=torch.load(entry['path'],weights_only=True);E=state['state']['encoder'].double().cuda();scale=state['scale'].double().cuda();W=E@D/scale
  for idx in indices:
   z=zs[idx];J=(W*(R@z)[None,:])@L+(W*(L@z)[None,:])@R
   T=J-(J@z)[:,None]*z[None,:]/z.square().sum();sv=torch.linalg.svdvals(T);rank=int((sv>sv[0]*p['rank_relative_tolerance']).sum());radial=float((T@z).norm()/max(float(T.norm()*z.norm()),1));assert radial<1e-12
   gen=torch.Generator(device='cuda').manual_seed(11300+idx);direction=torch.randn(z.shape,generator=gen,device='cuda',dtype=torch.float64);direction/=direction.norm()
   def encoded_on_sphere(t):
    u=z+t*direction;u=u/u.norm()*z.norm();return E@(D*((L@u)*(R@u))[None,:]).sum(1)/scale
   # Independent autograd through the native polynomial and explicit normalization.
   _,automatic=torch.autograd.functional.jvp(encoded_on_sphere,torch.zeros((),device='cuda',dtype=torch.float64),torch.ones((),device='cuda',dtype=torch.float64))
   predicted=T@direction;err=float((automatic-predicted).norm()/max(float(predicted.norm()),1));maxjvp=max(maxjvp,err);assert err<1e-10
   if idx in [indices[0],indices[-1]]:
    independent=np.linalg.svd(T.cpu().numpy(),compute_uv=False);discrepancy=float(np.max(np.abs(independent-sv.cpu().numpy()))/max(float(sv[0]),1));assert discrepancy<1e-10;svdaudits.append(dict(seed=entry['seed'],mode=entry['mode'],index=idx,relative_max_discrepancy=discrepancy))
   rows.append(dict(seed=entry['seed'],mode=entry['mode'],index=idx,numerical_rank=rank,min_singular=float(sv[-1]),max_singular=float(sv[0]),condition_number=float(sv[0]/sv[-1]),radial_relative_error=radial,jvp_relative_error=err))
 out=dict(status='completed',source_sha256=sha(__file__),protocol_sha256=sha(pp),rows=rows,independent_numpy_svd_checks=svdaudits,max_jvp_relative_error=maxjvp,scope='Local differential at16 fixed natural training states, tangent to constant-RMS sphere. Rank tests local freedom of raw encoder coordinates; no guarantee a nearby gate tie is reachable on a valid text manifold. No intervention or training.')
 dest=V/'native_code_tangent.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('ROWS',len(rows),'RANKS',sorted({x['numerical_rank'] for x in rows}),'MIN_SINGULAR_RANGE',min(x['min_singular'] for x in rows),max(x['min_singular'] for x in rows),'JVP',maxjvp)
if __name__=='__main__':main()

"""H562: local response lower bound in full training-context span; oracle diagnostic only."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_context_span_oracle'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);src=ART/'bilin18_response_weighted_fresh.json';bases=torch.load(ART/'bilin18_response_weighted_subspace_bases.pt',weights_only=False);oldinputs=torch.load(ART/'bilin18_mlp_response_decomposition_inputs.pt',weights_only=False);oldrows=torch.load(ART/'bilin18_position_vocabulary_control_native_cache.pt',weights_only=False)['rows'];fresh=torch.load(ART/'bilin18_response_weighted_fresh_inputs.pt',weights_only=False);rows=[dict(r,split='old_holdout') for r in oldrows if int(r['id'].rsplit('-',2)[1])>=4]+[dict(r,split='fresh') for r in fresh['rows']];inputs=dict(oldinputs,**fresh['inputs']);cfg=dict(experiment='H562',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),scale=.0625,design='All 39 training-span directions, ordinary projection and response-optimal oracle coordinates for each recipient displacement. Oracle uses exact target response and is NOT a predictor or validation result.',scope='Local L8 vector reconstruction only, no live downstream interventions or KL. Lower bound conditional on fixed training span, mean context, exact displacement and quadratic term.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda');m=model.blocks[8].mlp.float().eval();W=bases['W'].cuda();mean=bases['mean'].cuda();records=[]
 for r in rows:
  x=inputs[r['id']]['x'].cuda();z=F.rms_norm(x,(x.numel(),));e=F.rms_norm(x+.0625*inputs[r['id']]['delta'].cuda(),(x.numel(),))-z
  def lin(v):return m.Down(m.Left(v)*m.Right(e)+m.Left(e)*m.Right(v))
  A=lin(W).double().cpu();target=(lin(z)-lin(mean)).double().cpu();coords=((z-mean)@W.T).double().cpu();sol=torch.linalg.lstsq(A.T,target,driver='gelsd');oracle=sol.solution;den=(lin(z)+m.Down(m.Left(e)*m.Right(e))).norm().double().cpu();normal=(A@(target-A.T@oracle)).norm()/(A.norm()*target.norm());records.append(dict(id=r['id'],family=r['family'],split=r['split'],span_dimension=len(W),effective_response_rank=int(sol.rank),projection_relative_error=float((target-A.T@coords).norm()/den),oracle_relative_error=float((target-A.T@oracle).norm()/den),normal_equation_relative_residual=float(normal),context_residual_relative_norm=float((z-mean-((z-mean)@W.T)@W).norm()/(z-mean).norm())))
 result=dict(config=cfg,records=records,summary={},status='complete')
 for split in ('old_holdout','fresh'):
  result['summary'][split]={}
  for fam in sorted({r['family'] for r in rows}):
   cc=[c for c in records if c['split']==split and c['family']==fam];result['summary'][split][fam]={key:float(np.mean([c[key] for c in cc])) for key in ('projection_relative_error','oracle_relative_error','context_residual_relative_norm')}
 result['audits']=dict(oracle_never_worse=all(c['oracle_relative_error']<=c['projection_relative_error']+1e-6 for c in records),max_normal_residual=max(c['normal_equation_relative_residual'] for c in records),finite=all(np.isfinite(c['oracle_relative_error']) for c in records),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest());(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['summary'],indent=2));print(result['audits'])
if __name__=='__main__':main()

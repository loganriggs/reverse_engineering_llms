"""H557: transplant bilinear response operators, retaining recipient normalization change."""
import json,hashlib,time
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_mlp_context_operator'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_mlp_response_decomposition.json';inputs=torch.load(ART/'bilin18_mlp_response_decomposition_inputs.pt',weights_only=False);saved=torch.load(ART/'bilin18_position_vocabulary_control_native_cache.pt',weights_only=False);rows=saved['rows'];native=saved['native'];by={r['id']:r for r in rows};old=json.loads((ART/'bilin18_position_response_scale.json').read_text());cfg=dict(experiment='H557',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),scales=[-.0625,.0625],head=[8,3],layer=8,design='Reuse H554 panel. Change L8 bilinear linear-response operator by substituting donor normalized base context; retain recipient normalized displacement, its quadratic term, baseline MLP output and all downstream computation. Donor rules fixed before evaluation.',donors=['self','other_token','other_format','other_vocabulary','family_mean_leave_pair_out'],scope='Diagnostic operator transfer, not held-out learned canonical basis. Missing color-binding donors omitted explicitly. Does not transplant full donor activations or task answers.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 m=model.blocks[8].mlp;z={key:F.rms_norm(v['x'].cuda(),(v['x'].numel(),)) for key,v in inputs.items()};result=dict(config=cfg,geometry=[],interventions=[],omitted=[],status='running');jobs=defaultdict(list)
 for r in rows:
  fam,item,variant=r['id'].rsplit('-',2);otherfmt={'name_prose':'name_record','name_record':'name_prose','color_prose':'color_record','color_record':'color_prose','binding':'name_prose'}[fam];donors=dict(self=z[r['id']],other_token=z[f'{fam}-{item}-{1-int(variant)}'],other_format=z[f'{otherfmt}-{item}-{variant}'],family_mean_leave_pair_out=torch.stack([z[q['id']] for q in rows if q['family']==fam and q['pair']!=r['pair']]).mean(0))
  if fam!='binding':donors['other_vocabulary']=z[f"{fam.replace('name_','color_') if fam.startswith('name_') else fam.replace('color_','name_')}-{item}-{variant}"]
  else:result['omitted'].append(dict(id=r['id'],mode='other_vocabulary',reason='No color-binding counterpart'))
  x=inputs[r['id']]['x'].cuda();y0=m(z[r['id']])
  for scale in cfg['scales']:
   delta=scale*inputs[r['id']]['delta'].cuda();e=F.rms_norm(x+delta,(x.numel(),))-z[r['id']];quad=m.Down(m.Left(e)*m.Right(e));actual=m(z[r['id']]+e)-y0
   for mode,donor in donors.items():
    response=m.Down(m.Left(donor)*m.Right(e)+m.Left(e)*m.Right(donor))+quad;result['geometry'].append(dict(id=r['id'],family=fam,scale=scale,mode=mode,relative_response_error=float((response-actual).norm()/actual.norm()),response_cosine=float(F.cosine_similarity(response[None],actual[None])[0])));jobs[len(r['ids'])].append((r,scale,mode,delta.cpu(),(y0+response).cpu()))
 print('LOCKED JOBS',sum(map(len,jobs.values())),flush=True)
 for T,jj in jobs.items():
  for begin in range(0,len(jj),24):
   bb=jj[begin:begin+24]
   def hook(mod,args,out):out=out.clone();out[:,-1]=torch.stack([y for r,s,m,d,y in bb]).to(out);return out
   handle=m.register_forward_hook(hook)
   try:zz,_=run(model,torch.tensor([r['ids'] for r,s,m,d,y in bb],device='cuda'),patch=(8,torch.stack([d for r,s,m,d,y in bb])))
   finally:handle.remove()
   for ix,(r,scale,mode,d,y) in enumerate(bb):result['interventions'].append(dict(id=r['id'],family=r['family'],scale=scale,mode=mode,margin_delta=measure(zz[ix],r['groups'])['margin']-native[r['id']]['margin']))
 live={(c['id'],c['scale']):c['margin_delta'] for c in old['interventions'] if c['clamp']=='live'};result['summary']={}
 for fam in sorted({r['family'] for r in rows}):
  result['summary'][fam]={}
  for mode in cfg['donors']:
   cc=[c for c in result['interventions'] if c['family']==fam and c['mode']==mode];gg=[g for g in result['geometry'] if g['family']==fam and g['mode']==mode]
   if not cc:continue
   nz=[c for c in cc if abs(c['margin_delta'])>1e-4 and abs(live[c['id'],c['scale']])>1e-4];result['summary'][fam][mode]=dict(mean_relative_response_error=float(np.mean([g['relative_response_error'] for g in gg])),mean_response_cosine=float(np.mean([g['response_cosine'] for g in gg])),margin_mae=float(np.mean([abs(c['margin_delta']-live[c['id'],c['scale']]) for c in cc])),mean_absolute_native_effect=float(np.mean([abs(live[c['id'],c['scale']]) for c in cc])),sign_agreement=float(np.mean([np.sign(c['margin_delta'])==np.sign(live[c['id'],c['scale']]) for c in nz])) if nz else None,n_sign=len(nz))
 result['audits']=dict(self_live_replay_max_error=max(abs(c['margin_delta']-live[c['id'],c['scale']]) for c in result['interventions'] if c['mode']=='self'),finite=all(np.isfinite(c['margin_delta']) for c in result['interventions']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['audits']),flush=True)
if __name__=='__main__':main()

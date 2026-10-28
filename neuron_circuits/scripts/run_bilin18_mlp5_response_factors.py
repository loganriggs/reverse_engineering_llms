"""H626 analytic normalization/bilinear factor swaps for MLP5 tangent."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_mlp5_response_factors'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();mp=ART/'bilin18_emission_surface_transfer_manifest.json';vp=ART/'bilin18_emission_surface_transfer_locked.pt';cp=ART/'bilin18_emission_readout_trace_captures.pt';rows=json.loads(mp.read_text());v=torch.load(vp,weights_only=False)['common'].double();prior=torch.load(cp,weights_only=False);paths={'manifest':mp,'vector':vp,'trace':cp,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};train=[i for i,e in enumerate(rows) if e['item']<4];captrain=[i for i in train if rows[i]['family']=='capital' and rows[i]['form']==0];cells=sorted({(e['family'],e['form']) for e in rows});cfg=dict(experiment='H626',hashes=hashes,design='Native MLP5 raw residual and normalized input capture. Exact RMS derivative t=DN_x[v], ungated bilinear derivative B(y,t). Replace y or t with first4 capital/pool/cell discovery mean separately. Compare product of means and mean products; scalar norm and radial controls. No downstream fitting or model intervention.',scope='Local analytic response prediction, not an additive attribution or downstream causal rescue. Same adaptive72-prompt panel and fixed discovery indices.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 assert not model.config.gated
 buckets=defaultdict(list)
 for i,e in enumerate(rows):buckets[len(e['ids'])].append(i)
 X=torch.empty(len(rows),1152);Y=torch.empty_like(X);Z=torch.empty_like(prior['logits'][0.]);block=model.blocks[5]
 for ii in buckets.values():
  for st in range(0,len(ii),16):
   bb=ii[st:st+16];cap={};hs=[]
   def pre(m,args):cap['base']=(block.lambdas[0]*args[0][:,-1]+block.lambdas[1]*args[2][:,-1]).cpu()
   def att(m,args,out):cap['att']=out[0][:,-1].cpu()
   def mlp(m,args):cap['y']=args[0][:,-1].cpu()
   hs=[block.register_forward_pre_hook(pre),block.attn.register_forward_hook(att),block.mlp.register_forward_pre_hook(mlp)]
   try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'))
   finally:
    for h in hs:h.remove()
   X[bb]=cap['base']+cap['att'];Y[bb]=cap['y'];Z[bb]=z.cpu()
 X=X.double();r=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();y=X/r;t=v/r-X*(X*v).mean(-1,keepdim=True)/r.pow(3);mlp=block.mlp;L=mlp.Left.weight.double();R=mlp.Right.weight.double();D=mlp.Down.weight.double()
 def B(c,delta):
  c=c.cuda();delta=delta.cuda();return (((c@L.T)*(delta@R.T)+(delta@L.T)*(c@R.T))@D.T).cpu()
 exact=B(y,t);pred={'native':exact,'no_radial':B(y,v.expand_as(y))/r};fits={}
 for name,ii in [('capital',captrain),('pooled',train)]+[(f'{f}_{form}',[i for i in train if rows[i]['family']==f and rows[i]['form']==form]) for f,form in cells]:fits[name]=dict(y=y[ii].mean(0),t=t[ii].mean(0),mean_response=exact[ii].mean(0))
 for name in ['capital','pooled']:
  cy=fits[name]['y'].expand_as(y);ct=fits[name]['t'].expand_as(t);pred[name+'_context_native_norm']=B(cy,t);pred[name+'_norm_native_context']=B(y,ct);pred[name+'_both']=B(cy,ct);pred[name+'_mean_response']=fits[name]['mean_response'].expand_as(exact);pred[name+'_scalar_rms']=B(cy,v.expand_as(y))/r
 cy=torch.empty_like(y);ct=torch.empty_like(t)
 for f,form in cells:
  ii=[i for i,e in enumerate(rows) if e['family']==f and e['form']==form];cy[ii]=fits[f'{f}_{form}']['y'];ct[ii]=fits[f'{f}_{form}']['t']
 pred['cell_context_native_norm']=B(cy,t);pred['cell_norm_native_context']=B(y,ct);pred['cell_both']=B(cy,ct)
 # Bilinear expansion around capital mean is exact, including crossed factor deviations.
 y0=fits['capital']['y'].expand_as(y);t0=fits['capital']['t'].expand_as(t);base=B(y0,t0);yc=B(y-y0,t0);tc=B(y0,t-t0);cross=B(y-y0,t-t0);decomp_error=float(((base+yc+tc+cross-exact).norm(dim=-1)/exact.norm(dim=-1)).max());records=[];geometry=[];target=prior['tangents']['mlp5'];total_target=prior['tangents']['block5']
 for i,e in enumerate(rows):
  common=dict(id=e['id'],family=e['family'],form=e['form'],split='fit' if i in train else 'holdout');geometry.append(dict(**common,rms=float(r[i,0]),normalization_tangent_norm=float(t[i].norm()),norm_tangent_capital_cosine=float(torch.nn.functional.cosine_similarity(t[i],fits['capital']['t'],dim=0)),normalized_context_capital_cosine=float(torch.nn.functional.cosine_similarity(y[i],fits['capital']['y'],dim=0)),radial_norm_fraction=float((t[i]-v/r[i]).norm()/t[i].norm()),context_deviation_norm=float(yc[i].norm()),normalization_deviation_norm=float(tc[i].norm()),cross_deviation_norm=float(cross[i].norm()),native_response_norm=float(exact[i].norm())))
  for name,p in pred.items():records.append(dict(**common,mode=name,mlp_relative_error=float((p[i]-target[i]).norm()/target[i].norm()),block_relative_error=float((v+p[i]-total_target[i]).norm()/total_target[i].norm()),cosine=float(torch.nn.functional.cosine_similarity(p[i],target[i],dim=0))))
 result=dict(config=cfg,records=records,geometry=geometry,summary={},geometry_summary={},status='complete')
 for f,form in cells:
  for split in ['fit','holdout']:
   for name in pred:
    rr=[a for a in records if a['family']==f and a['form']==form and a['split']==split and a['mode']==name];result['summary'][f'{f}_{form}_{split}_{name}']={n:float(np.mean([a[n] for a in rr])) for n in ['mlp_relative_error','block_relative_error','cosine']}
   rr=[a for a in geometry if a['family']==f and a['form']==form and a['split']==split];result['geometry_summary'][f'{f}_{form}_{split}']={n:float(np.mean([a[n] for a in rr])) for n in ['rms','normalization_tangent_norm','norm_tangent_capital_cosine','normalized_context_capital_cosine','radial_norm_fraction','context_deviation_norm','normalization_deviation_norm','cross_deviation_norm','native_response_norm']}
 result['audits']=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},native_forwards=len(rows),native_logit_replay_max_abs=float((Z-prior['logits'][0.]).abs().max()),normalized_input_replay_max_abs=float((y-Y.double()).abs().max()),analytic_mlp_response_max_relative_error=max(a['mlp_relative_error'] for a in records if a['mode']=='native'),analytic_block_response_max_relative_error=max(a['block_relative_error'] for a in records if a['mode']=='native'),factor_expansion_max_relative_error=decomp_error,finite=all(np.isfinite(a['mlp_relative_error']) for a in records));result['elapsed_seconds']=time.time()-start;torch.save(dict(raw=X,normalized=y,normalization_tangent=t,exact=exact,predictions=pred,fits=fits,decomposition=dict(base=base,context=yc,normalization=tc,cross=cross)),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()

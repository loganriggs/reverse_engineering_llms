"""H627 causal local-tangent replacement with native baselines preserved."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_mlp5_response_replace'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();mp=ART/'bilin18_emission_surface_transfer_manifest.json';vp=ART/'bilin18_emission_surface_transfer_locked.pt';tp=ART/'bilin18_emission_readout_trace_captures.pt';fp=ART/'bilin18_mlp5_response_factors_captures.pt';rows=json.loads(mp.read_text());v=torch.load(vp,weights_only=False)['common'];trace=torch.load(tp,weights_only=False);f=torch.load(fp,weights_only=False);native_mlp=trace['captures'][0.]['mlp5'];responses={'analytic':f['exact'],'shared_context':f['predictions']['capital_context_native_norm'],'shared_normalization':f['predictions']['capital_norm_native_context'],'shared_both':f['predictions']['capital_both'],'zero_mlp_response':torch.zeros_like(f['exact'])};paths={'manifest':mp,'vector':vp,'trace':tp,'factors':fp,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};modes=['live']+list(responses);alphas=[-.02,0.,.02];cfg=dict(experiment='H627',hashes=hashes,modes=modes,alphas=alphas,design='Add alpha*v at final Attention5 output. Replace final MLP5 write with cached native write +alpha*specified analytic response. Native zero-alpha baseline remains recipient-specific; downstream live. Native analytic, shared-context, shared-normalization, both-shared and zero-response controls.',scope='Local derivative intervention, not finite-dose native computation or additive layer share. Same72 adaptive prompts, capital first4 factor fits; heldout item4-7 analysis.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 buckets=defaultdict(list)
 for i,e in enumerate(rows):buckets[len(e['ids'])].append(i)
 outputs={};writes={};finals={};baseline_error=[]
 for mode in modes:
  outputs[mode]={};writes[mode]={};finals[mode]={}
  for alpha in alphas:
   Z=torch.empty_like(trace['logits'][0.]);M=torch.empty_like(native_mlp);X=torch.empty_like(native_mlp)
   for ii in buckets.values():
    for st in range(0,len(ii),16):
     bb=ii[st:st+16];cap={};hs=[]
     def editatt(m,args,out):y,b=out;y=y.clone();y[:,-1]+=alpha*v.to(y);return y,b
     hs.append(model.blocks[5].attn.register_forward_hook(editatt))
     if mode!='live':
      replacement=native_mlp[bb].double()+alpha*responses[mode][bb]
      def editmlp(m,args,out):y=out.clone();y[:,-1]=replacement.to(y);return y
      hs.append(model.blocks[5].mlp.register_forward_hook(editmlp))
     def capmlp(m,args,out):cap['m']=out[:,-1].cpu()
     def capfinal(m,args,out):cap['x']=out[0][:,-1].cpu()
     hs.extend([model.blocks[5].mlp.register_forward_hook(capmlp),model.blocks[17].register_forward_hook(capfinal)])
     try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'))
     finally:
      for h in hs:h.remove()
     Z[bb]=z.cpu();M[bb]=cap['m'];X[bb]=cap['x']
   outputs[mode][alpha]=Z;writes[mode][alpha]=M;finals[mode][alpha]=X
   if alpha==0:baseline_error.append(float((Z-trace['logits'][0.]).abs().max()))
  print('COMPLETE',mode,flush=True)
 tangents={m:(outputs[m][.02].double()-outputs[m][-.02].double())/.04 for m in modes};dx={m:(finals[m][.02].double()-finals[m][-.02].double())/.04 for m in modes};captrain=[i for i,e in enumerate(rows) if e['family']=='capital' and e['form']==0 and e['item']<4];means={m:tangents[m][captrain].mean(0) for m in modes};records=[];patcherrs=[]
 for mode in responses:
  got=(writes[mode][.02].double()-writes[mode][-.02].double())/.04;patcherrs.append(float(((got-responses[mode]).norm(dim=-1)/responses[mode].norm(dim=-1).clamp_min(1.)).max()))
 for i,e in enumerate(rows):
  p=outputs['live'][0.][i].double().softmax(0);g=tangents['live'][i];ans=e['variants'][e['item']][0][0]
  def center(x):return x-(p*x).sum()
  def norm(x):return (center(x).square()*p).sum().sqrt()
  for mode in modes:
   h=tangents[mode][i];mean=means[mode];gain=float((center(h)*center(mean)*p).sum()/(center(mean).square()*p).sum());records.append(dict(id=e['id'],family=e['family'],form=e['form'],split='fit' if e['item']<4 else 'holdout',mode=mode,response_change=float(norm(h-g)/norm(g)),response_norm_ratio=float(norm(h)/norm(g)),final_residual_change=float((dx[mode][i]-dx['live'][i]).norm()/dx['live'][i].norm()),own_mode_capital_cosine=float((center(h)*center(mean)*p).sum()/(norm(h)*norm(mean))),own_mode_capital_error=float(norm(h-mean)/norm(h)),own_mode_scalar_error=float(norm(h-gain*mean)/norm(h)),answer_logprob_derivative=float(center(h)[ans])))
 result=dict(config=cfg,records=records,summary={},status='complete')
 for family,form in sorted({(e['family'],e['form']) for e in rows}):
  for split in ['fit','holdout']:
   for mode in modes:
    rr=[r for r in records if r['family']==family and r['form']==form and r['split']==split and r['mode']==mode];result['summary'][f'{family}_{form}_{split}_{mode}']={n:float(np.mean([r[n] for r in rr])) for n in ['response_change','response_norm_ratio','final_residual_change','own_mode_capital_cosine','own_mode_capital_error','own_mode_scalar_error','answer_logprob_derivative']}
 result['audits']=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},evaluations=len(rows)*len(modes)*len(alphas),all_mode_zero_alpha_max_logit_error=max(baseline_error),live_prior_max_logit_error=max(float((outputs['live'][alpha]-trace['logits'][alpha]).abs().max()) for alpha in alphas),replacement_response_max_relative_error=max(patcherrs),analytic_output_response_max_error=max(r['response_change'] for r in records if r['mode']=='analytic'),finite=all(np.isfinite(r['response_change']) for r in records));torch.save(dict(outputs=outputs,logit_tangents=tangents,final_tangents=dx),ART/f'{STEM}_captures.pt');result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()

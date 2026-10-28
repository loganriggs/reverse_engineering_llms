"""H375 selected-case finite-tail tracing and finite-KL oracle."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_oracle_tail_mechanism'
def main():
 torch.set_grad_enabled(False);torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 pre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]]
 x=F.rms_norm(pre,(1152,));state=pre+B(x,x,L,R,D)+model.blocks[17].mlp.Down_bias;baseline=F.rms_norm(state,(1152,))
 W=model.lm_head.weight.detach().double();assert model.lm_head.bias is None
 from run_bilin18_qkov_causal_edges import TOKENIZER,AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);tokens=np.load(ROOT/cfg['corpus']);V=torch.tensor(np.load(ART/'bilin18_l17_probability_axes_finite_frozen_basis.npz')['eigenvectors'][:,:16],device='cuda');mapped=V.T@W.T;old=np.load(ART/'bilin18_l17_probability_axes_oracle_rows.npz');responses=np.load(ART/'bilin18_l17_shared_geometry_precision_h6_responses.npz')['responses'];results=[];saved={};replay=0.
 def lp(raw):return (30*torch.tanh(raw/30)).log_softmax(-1)
 def divergence(t,p):return (t.exp()*(t-p)).sum()
 for case in cfg['cases']:
  ci=case['condition'];si=case['site'];response=torch.tensor(responses[ci,si],device='cuda');target=F.rms_norm(state[si]+response,(1152,));delta=(target-baseline[si]).double();raw=baseline[si].double()@W.T;dlog=delta@W.T;coef=torch.tensor(old['oracle_coefficients'][26+ci,0,si,:16],device='cuda');edit=coef@mapped;base=lp(raw);fulltarget=lp(raw+dlog);rare=base.exp()<cfg['rare_probability'];cap=1-torch.tanh(raw/30).square();basec=30*torch.tanh(raw/30);trace=[]
  for t in cfg['amplitudes']:
   targetlp=lp(raw+t*dlog);patchlp=lp(raw+t*edit);lin_target=(basec+t*cap*dlog).log_softmax(-1);lin_patch=(basec+t*cap*edit).log_softmax(-1)
   trace.append({'amplitude':t,'baseline_kl':float(divergence(targetlp,base)),'oracle16_kl':float(divergence(targetlp,patchlp)),'linearized_cap_kl':float(divergence(lin_target,lin_patch)),'rare_mass_target':float(targetlp.exp()[rare].sum()),'rare_mass_patch':float(patchlp.exp()[rare].sum()),'top_patch_token':int(patchlp.argmax()),'top_patch_probability':float(patchlp.exp().max())})
  replay=max(replay,abs(trace[-1]['oracle16_kl']-float(old['kl'][2,ci,3,si]))/max(trace[-1]['oracle16_kl'],1e-12));patch=lp(raw+edit);top=torch.topk(patch,5).indices;toprows=[]
  for idx in top.tolist():toprows.append({'id':idx,'text':tokenizer.decode([idx]),'baseline_probability':float(base[idx].exp()),'target_probability':float(fulltarget[idx].exp()),'oracle16_probability':float(patch[idx].exp()),'raw_logit_delta':float(edit[idx]),'capped_logit_delta':float((30*torch.tanh((raw[idx]+edit[idx])/30))-basec[idx])})
  fits=[];best=float('inf');bestcoef=None
  for name,start in [('zero',torch.zeros_like(coef)),('euclidean',delta@V),('local_oracle',coef)]:
   with torch.enable_grad():
    param=start.detach().clone().requires_grad_(True);opt=torch.optim.LBFGS([param],lr=1.,max_iter=cfg['fit_max_iter'],tolerance_grad=1e-10,tolerance_change=1e-13,line_search_fn='strong_wolfe');history=[]
    def closure():
     opt.zero_grad();loss=divergence(fulltarget,lp(raw+param@mapped));loss.backward();history.append(float(loss.detach()));return loss
    opt.step(closure);loss=closure();value=float(loss.detach());gradient=float(param.grad.norm());cc=param.detach().clone()
   fits.append({'start':name,'kl':value,'gradient_norm':gradient,'evaluations':len(history),'loss_history':history})
   if value<best:best=value;bestcoef=cc
  saved[case['name']]=bestcoef.cpu().numpy();doc,pos=divmod(si,4);position=cfg['positions'][pos];context=tokenizer.decode(tokens[doc,max(0,position-25):position+1].tolist());results.append({'case':case,'context':context,'trace':trace,'top_oracle16_tokens':toprows,'finite_fits':fits,'best_finite_kl':best,'best_finite_recovery':1-best/trace[-1]['baseline_kl'],'best_coefficient_norm':float(bestcoef.norm()),'local_coefficient_norm':float(coef.norm())});print(case['name'],'old',trace[-1]['oracle16_kl'],'fit',best,flush=True)
 assert replay<1e-7
 tails=[x for x in results if x['case']['name'] in ['moment_site1007','donor_site1007']];gates={'rare_token_takeover':all(x['trace'][-1]['rare_mass_patch']>.5 for x in tails),'finite_fit_repairs_tail':all(x['best_finite_kl']<=.01*x['trace'][-1]['oracle16_kl'] for x in tails),'finite_fit_beats_baseline':all(x['best_finite_kl']<x['trace'][-1]['baseline_kl'] for x in tails)}
 np.savez_compressed(STEM.with_name(STEM.name+'_finite_coefficients.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'audit':{'h374_selected_kl_replay_relative':replay},'gates':gates},indent=2)+'\n');print('FINISHED',gates,flush=True)
if __name__=='__main__':main()

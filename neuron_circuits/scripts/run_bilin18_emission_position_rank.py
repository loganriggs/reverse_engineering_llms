"""H622 positional and rank dissection of fixed emission intervention."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_emission_position_rank'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);vp=ART/'bilin18_frozen_emission_direction_locked.pt';mp=ART/'bilin18_frozen_emission_direction_manifest.json';V=torch.load(vp,weights_only=False)['vectors']['common'];U,S,W=torch.linalg.svd(V.double(),full_matrices=False);vec={'full':V}
 for i in range(6):
  x=torch.zeros_like(V);x[i]=V[i];vec[f'row{i}']=x
 for name,sl in [('first5',slice(0,5)),('last2',slice(4,6)),('first4',slice(0,4))]:
  x=torch.zeros_like(V);x[sl]=V[sl];vec[name]=x
 for r in [1,2]:vec[f'rank{r}']=((U[:,:r]*S[:r])@W[:r]).float()
 locked=ART/f'{STEM}_locked.pt';torch.save(vec,locked);paths={'source':vp,'manifest':mp,'vectors':locked,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};rows=[e for e in json.loads(mp.read_text()) if e['prefix'] and e['lower'] and e['family'] in ['capital0','capital1','language']];doses=[-1.,.25,1.];cfg=dict(experiment='H622',hashes=hashes,doses=doses,modes=list(vec),design='Frozen H621 six-position vector. Individual rows, first5,last2,first4 and unscaled rank1/2 SVD approximations; subtraction positive/addition negative. Native and full-removal references. Logit factorial comparisons separate from KL. No downstream fit or rescaling.',scope='Adaptive capital/language panel,24 recipients. SVD derived only from frozen source. Same token positions; no new task or scaffold validation.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 def forward(delta=None):
  cap={};hs=[]
  def pre(m,args):cap['t']=args[0][:,-1].cpu()
  hs.append(model.blocks[11].mlp.register_forward_pre_hook(pre))
  if delta is not None:
   def edit(m,args,out):y,b=out;y=y.clone();y[:,4:]-=delta.to(y);return y,b
   hs.append(model.blocks[5].attn.register_forward_hook(edit))
  try:z,_=run(model,torch.tensor([e['ids'] for e in rows],device='cuda'))
  finally:
   for h in hs:h.remove()
  return z.cpu(),cap['t']
 nz,nt=forward();result=dict(config=cfg,records=[],factorial=[],summary={},factorial_summary={},status='running');generic=[tok.encode(s,add_special_tokens=False)[0] for s in [' the',' a',' not']]
 def kl(a,b):a=a.double().log_softmax(0);b=b.double().log_softmax(0);return float((a.exp()*(a-b)).sum())
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 for dose in doses:
  outputs={}
  for mode,x in vec.items():
   zz,tt=forward(x*dose);outputs[mode]=(zz,tt);fz,ft=outputs['full']
   for i,e in enumerate(rows):
    z=zz[i];lp=z.double().log_softmax(0);ans=e['answer_id'];cs=e['candidates'];result['records'].append(dict(id=e['id'],family=e['family'],split=e['split'],mode=mode,dose=dose,kl=kl(nz[i],z),reference_kl=kl(fz[i],z),omission_reference_kl=kl(fz[i],nz[i]),target_error=float((tt[i]-ft[i]).norm()/(ft[i]-nt[i]).norm()),top1_correct=int(z.argmax())==ans,candidate_correct=cs[int(z[cs].argmax())]==ans,answer_mass=float(lp[ans].exp()),answer_ce=float(-lp[ans]),generic_margin=float(z[ans].double()-torch.logsumexp(z[generic].double(),0)),top_token=tok.decode([int(z.argmax())])))
  for i,e in enumerate(rows):
   fz,ft=[x[i] for x in outputs['full']];prob=nz[i].double().softmax(0)
   def norm(x):x=x.double();return ((x-(prob*x).sum()).square()*prob).sum().sqrt()
   for name,modes in [('six_rows',[f'row{r}' for r in range(6)]),('first5_last',['first5','row5']),('first4_last2',['first4','last2'])]:
    predz=nz[i]+sum(outputs[m][0][i]-nz[i] for m in modes);predt=nt[i]+sum(outputs[m][1][i]-nt[i] for m in modes);result['factorial'].append(dict(id=e['id'],family=e['family'],dose=dose,decomposition=name,logit_cross_relative=float(norm(fz-predz)/norm(fz-nz[i])),target_cross_relative=float((ft-predt).norm()/(ft-nt[i]).norm()),additive_prediction_kl=kl(fz,predz)))
  save();print('COMPLETE',dose,flush=True)
 for family in ['capital0','capital1','language']:
  for dose in doses:
   for mode in vec:
    rr=[r for r in result['records'] if r['family']==family and r['dose']==dose and r['mode']==mode];out={n:float(np.mean([r[n] for r in rr])) for n in ['kl','reference_kl','target_error','top1_correct','candidate_correct','answer_mass','generic_margin','answer_ce']};out['relative_reference_kl']=out['reference_kl']/np.mean([r['omission_reference_kl'] for r in rr]);result['summary'][f'{family}_{dose}_{mode}']=out
   for decomp in ['six_rows','first5_last','first4_last2']:
    rr=[r for r in result['factorial'] if r['family']==family and r['dose']==dose and r['decomposition']==decomp];result['factorial_summary'][f'{family}_{dose}_{decomp}']={n:float(np.mean([r[n] for r in rr])) for n in ['logit_cross_relative','target_cross_relative','additive_prediction_kl']}
 result['audits']=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},native_prompts=len(rows),evaluations=len(result['records']),factorial_cases=len(result['factorial']),row_sum_max_abs=float((sum(vec[f'row{i}'] for i in range(6))-V).abs().max()),svd_rank_energy={r:float(S[:r].square().sum()/S.square().sum()) for r in [1,2]},finite=all(np.isfinite(r[n]) for r in result['records'] for n in ['kl','reference_kl','target_error']));result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()

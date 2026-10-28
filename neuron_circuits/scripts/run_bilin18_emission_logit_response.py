"""H624 small-dose vocabulary response and frozen response prediction."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_emission_logit_response'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);mp=ART/'bilin18_emission_surface_transfer_manifest.json';vp=ART/'bilin18_emission_surface_transfer_locked.pt';rows=json.loads(mp.read_text());v=torch.load(vp,weights_only=False)['common'];paths={'manifest':mp,'vector':vp,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};alphas=[-1.,-.25,-.05,-.02,0.,.02,.05,.25,1.];cfg=dict(experiment='H624',hashes=hashes,alphas=alphas,design='Addition-signed alpha, frozen final Attention5 vector. Central logit differences at .02 and .05. Capital0 items0-3 mean, pooled first4 per cell mean, and per-cell first4 mean response fits. Test items4-7; local tangent and oracle scalar gain are diagnostic controls, not predictive models. Full vocabulary probability-weighted centered metrics and finite-dose KL.',scope='Adaptive same72 prompts, fit indices declared before inference. Context fits use same subject identities across forms; no claims of independent task generalization. Plural candidate audit adds singular competitors.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 buckets=defaultdict(list)
 for i,e in enumerate(rows):buckets[len(e['ids'])].append(i)
 logits={a:torch.empty(len(rows),model.lm_head.weight.shape[0]) for a in alphas}
 for alpha in alphas:
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16]
    def hook(m,args,out):y,b=out;y=y.clone();y[:,-1]+=alpha*v.to(y);return y,b
    h=model.blocks[5].attn.register_forward_hook(hook)
    try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'))
    finally:h.remove()
    logits[alpha][bb]=z.cpu()
  print('COMPLETE',alpha,flush=True)
 tangent=(logits[.02].double()-logits[-.02].double())/.04;coarse=(logits[.05].double()-logits[-.05].double())/.1;tangent-=tangent.mean(-1,keepdim=True);coarse-=coarse.mean(-1,keepdim=True);train=[i for i,e in enumerate(rows) if e['item']<4];cap=[i for i in train if rows[i]['family']=='capital' and rows[i]['form']==0];cells=sorted({(e['family'],e['form']) for e in rows});fits={'capital':tangent[cap].mean(0),'pooled':tangent[train].mean(0)}
 for family,form in cells:fits[f'{family}_{form}']=tangent[[i for i in train if rows[i]['family']==family and rows[i]['form']==form]].mean(0)
 locked=ART/f'{STEM}_locked.pt';torch.save(dict(fits=fits,train_ids=[rows[i]['id'] for i in train],capital_fit_ids=[rows[i]['id'] for i in cap]),locked);fh=hashlib.sha256(locked.read_bytes()).hexdigest();result=dict(config=cfg,geometry=[],predictions=[],summary={},geometry_summary={},native_competence={},status='running')
 def kl(a,b):a=a.double().log_softmax(0);b=b.double().log_softmax(0);return float((a.exp()*(a-b)).sum())
 for i,e in enumerate(rows):
  z=logits[0.][i].double();p=z.softmax(0);g=tangent[i]
  def center(x):return x-(p*x).sum()
  def norm(x):return ((center(x)**2*p).sum()).sqrt()
  def inner(x,y):return (center(x)*center(y)*p).sum()
  common=fits['capital'];gain=float(inner(g,common)/inner(common,common));ans=e['variants'][e['item']][0];assert len(ans)==1;ans=ans[0];gold=[vv[0][0] for vv in e['variants']];competitors=gold.copy()
  if e['family']=='plural':
   for subject in 'cat dog book car tree house bird boy'.split():
    ids=tok.encode(' '+subject,add_special_tokens=False);assert len(ids)==1;competitors+=ids
  result['geometry'].append(dict(id=e['id'],family=e['family'],form=e['form'],split='fit' if i in train else 'holdout',tangent_norm=float(norm(g)),step_check_relative=float(norm(g-coarse[i])/norm(g)),capital_cosine=float(inner(g,common)/(norm(g)*norm(common))),capital_error=float(norm(g-common)/norm(g)),pooled_error=float(norm(g-fits['pooled'])/norm(g)),cell_error=float(norm(g-fits[f"{e['family']}_{e['form']}"])/norm(g)),scalar_error=float(norm(g-gain*common)/norm(g)),oracle_gain=gain,answer_logprob_derivative=float(center(g)[ans]),capital_answer_derivative=float(center(common)[ans]),plausible_choice_correct=competitors[int(z[competitors].argmax())]==ans,declared_choice_correct=gold[int(z[gold].argmax())]==ans))
  for kind,response in [('capital',common),('pooled',fits['pooled']),('cell',fits[f"{e['family']}_{e['form']}"]),('oracle_scalar',gain*common),('local',g)]:
   for alpha in [-1.,-.25,.05,.25,1.]:
    actual=logits[alpha][i].double();prediction=z+alpha*response;result['predictions'].append(dict(id=e['id'],family=e['family'],form=e['form'],split='fit' if i in train else 'holdout',kind=kind,alpha=alpha,prediction_kl=kl(actual,prediction),omission_kl=kl(actual,z),answer_logprob_error=float(prediction.log_softmax(0)[ans]-actual.log_softmax(0)[ans])))
 for family,form in cells:
  for split in ['fit','holdout']:
   rr=[r for r in result['geometry'] if r['family']==family and r['form']==form and r['split']==split];result['geometry_summary'][f'{family}_{form}_{split}']={n:float(np.mean([r[n] for r in rr])) for n in ['tangent_norm','step_check_relative','capital_cosine','capital_error','pooled_error','cell_error','scalar_error','oracle_gain','answer_logprob_derivative','capital_answer_derivative','plausible_choice_correct','declared_choice_correct']}
   for kind in ['capital','pooled','cell','oracle_scalar','local']:
    for alpha in [-1.,-.25,.05,.25,1.]:
     rr=[r for r in result['predictions'] if r['family']==family and r['form']==form and r['split']==split and r['kind']==kind and r['alpha']==alpha];err=float(np.mean([r['prediction_kl'] for r in rr]));den=float(np.mean([r['omission_kl'] for r in rr]));result['summary'][f'{family}_{form}_{split}_{kind}_{alpha}']=dict(prediction_kl=err,omission_kl=den,relative_kl=err/den,mean_abs_answer_logprob_error=float(np.mean([abs(r['answer_logprob_error']) for r in rr])))
 result['top_common_tokens']={name:dict(increased=[tok.decode([i]) for i in vec.topk(15).indices.tolist()],decreased=[tok.decode([i]) for i in (-vec).topk(15).indices.tolist()]) for name,vec in fits.items() if name in ['capital','pooled']};result['audits']=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},fits_unchanged=fh==hashlib.sha256(locked.read_bytes()).hexdigest(),native_prompts=len(rows),evaluations=len(rows)*len(alphas),fit_examples=len(train),capital_fit_examples=len(cap),finite=all(np.isfinite(r['prediction_kl']) for r in result['predictions']));torch.save(dict(logits=logits,tangent=tangent.float(),coarse_tangent=coarse.float()),ART/f'{STEM}_captures.pt');result['elapsed_seconds']=time.time()-start;result['status']='complete';(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()

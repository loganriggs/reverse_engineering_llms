"""H765 actual natural-validation KL/CE for frozen sensitivity-weighted geometry."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_l5_natural_featurizer_baselines import reconstruct
from run_bilin18_verified_head_sparse import SignedBSF
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_geometry_output_fidelity'
def geometry_reconstruct(x,fit,k,adaptive=False):
 z=(x.double()-fit['mean'])@fit['encode']
 if adaptive:
  mask=torch.zeros_like(z).scatter_(1,z.abs().topk(k,dim=1).indices,1);z=z*mask
 else:z=z.clone();z[:,k:]=0
 return fit['mean']+z@fit['decode']
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_l5_natural_featurizer_baselines_data.pt','base_fits':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','fisher_fits':ART/'bilin18_l5_refined_output_geometry_captures.pt','scalar_fits':ART/'bilin18_l5_sparse_featurizers_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py')};hashes={k:sha(p) for k,p in paths.items()};docs=sorted(range(2048,2304),key=lambda d:hashlib.sha256(f'H765:{d}'.encode()).hexdigest())[:64];contexts=[{'doc':d,'length':n} for d in docs for n in [16,32,64,120]];cfg={'experiment':'H765','hashes':hashes,'contexts':contexts,'budgets':[8,16,32,64,128],'design':'Fixed64natural validation documents crossed with4prefix lengths,256contexts. All frozen H764 ridge settings, fixed and adaptive coordinate selection; H754fixed/adaptivewritePCA and3randomTopK controls, H755scalar8 all3seeds, native control. Replace final L5H5 head activation by reconstruction, downstream live. Measure actual KL(native||reconstructed) and observed next-token CE change separately, with write reconstruction error. No selection or fitting on these metrics.','limitations':'Reused validation documents, four contexts share each document. Not reserved-test evaluation. Local Fisher metric estimated on training contexts, not these outcomes. Fixed rank and adaptive active-coordinate budget different capacities; scalar has learned overcomplete supports. Low KL not semantic unit evidence.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['data'],weights_only=False);ids=data['all_token_ids'];baseline=torch.load(paths['base_fits'],weights_only=False)['fits']['L5H5'];fisher=torch.load(paths['fisher_fits'],weights_only=False)['fits'];scalar=torch.load(paths['scalar_fits'],weights_only=False)['states'];model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native={};heads={};buckets=defaultdict(list);forwards=0;captureerrors=[]
 for c in contexts:buckets[c['length']].append(c)
 for length,cc in buckets.items():
  for lo in range(0,len(cc),8):
   bb=cc[lo:lo+8];saved={}
   def cap(m,args):saved['head']=args[0][:,-1,640:768].cpu()
   hook=model.blocks[5].attn.c_proj.register_forward_pre_hook(cap)
   try:logits,_=run(model,ids[[c['doc'] for c in bb],:length].cuda())
   finally:hook.remove()
   forwards+=len(bb)
   for i,c in enumerate(bb):
    key=(c['doc'],length);native[key]=logits[i].cpu();heads[key]=saved['head'][i];captureerrors.append(float((heads[key]-data['activations']['L5H5'][c['doc'],length//8-1]).abs().max()))
 specs={'native':{'method':'native','budget':128,'seed':None}}
 for k in cfg['budgets']:
  for method in ['write_pca','adaptive_write_pca']:specs[f'{method}_{k}']={'method':method,'budget':k,'seed':None}
  for seed in [7541,7542,7543]:specs[f'random_{seed}_{k}']={'method':'random_topk','budget':k,'seed':seed}
  for f in fisher:
   for adaptive in [False,True]:specs[f'{f}_{"adaptive" if adaptive else "fixed"}_{k}']={'method':'fisher_adaptive' if adaptive else 'fisher_fixed','fit':f,'budget':k,'seed':None}
 for seed in [7551,7552,7553]:specs[f'scalar_{seed}']={'method':'scalar','budget':8,'seed':seed}
 keys=list(heads);raw=torch.stack([heads[k] for k in keys]).double();recons={}
 for name,s in specs.items():
  method=s['method'];k=s['budget']
  if method=='native':rec=raw
  elif method.startswith('fisher'):rec=geometry_reconstruct(raw,fisher[s['fit']],k,method=='fisher_adaptive')
  elif method=='scalar':
   st=scalar['L5H5',1,s['seed']];m=SignedBSF(1);m.load_state_dict(st['model']);x=((raw-baseline['mean'])@baseline['write_pca']['encode']/st['input_scale']).float();rec=baseline['mean']+(m(x).double()*st['input_scale'])@baseline['write_pca']['decode']
  else:rec=reconstruct(raw,baseline,method,k,s['seed'])
  recons[name]={key:rec[i].float() for i,key in enumerate(keys)}
 jobs=defaultdict(list)
 for name in specs:
  for c in contexts:jobs[c['length']].append((name,c))
 records=[];fullerrors=[];finite=True;W=baseline['write']
 for length,jj in jobs.items():
  for lo in range(0,len(jj),8):
   bb=jj[lo:lo+8];replacement=torch.stack([recons[n][c['doc'],length] for n,c in bb]).cuda()
   def edit(m,args):y=args[0].clone();y[:,-1,640:768]=replacement.to(y);return(y,)
   hook=model.blocks[5].attn.c_proj.register_forward_pre_hook(edit)
   try:logits,_=run(model,ids[[c['doc'] for n,c in bb],:length].cuda())
   finally:hook.remove()
   forwards+=len(bb);finite=finite and bool(torch.isfinite(logits).all())
   for i,(name,c) in enumerate(bb):
    key=c['doc'],length;p=logits[i].cpu();n=native[key];lp=p.double().log_softmax(-1);ln=n.double().log_softmax(-1);token=int(ids[c['doc'],length]);err=recons[name][key].double()-heads[key].double();records.append({**c,'variant':name,**specs[name],'native_to_reconstruction_KL':float((ln.exp()*(ln-lp)).sum()),'observed_token_CE_change':float(ln[token]-lp[token]),'native_observed_token_CE':float(-ln[token]),'write_error_squared':float((err@W.T).square().sum())})
    if specs[name]['budget']==128:fullerrors.append(float((p-n).abs().max()))
 groups=defaultdict(list)
 for r in records:groups[r['variant']].append(r)
 summary=[]
 for name,rr in groups.items():
  kl=torch.tensor([r['native_to_reconstruction_KL'] for r in rr],dtype=torch.double);ce=torch.tensor([r['observed_token_CE_change'] for r in rr],dtype=torch.double);docmeans=torch.tensor([sum(r['observed_token_CE_change'] for r in rr if r['doc']==d)/4 for d in docs]);summary.append({'variant':name,**specs[name],'mean_KL':float(kl.mean()),'median_KL':float(kl.median()),'p95_KL':float(torch.quantile(kl,.95)),'mean_CE_change':float(ce.mean()),'document_mean_CE_standard_error':float(docmeans.std()/len(docs)**.5),'mean_write_error_squared':sum(r['write_error_squared'] for r in rr)/len(rr)})
 audits={'hashes_unchanged':{k:sha(p)==hashes[k] for k,p in paths.items()},'validation_only':all(2048<=d<2304 for d in docs),'cached_capture_max_abs_error':max(captureerrors),'full_rank_and_native_logit_error_max_abs':max(fullerrors),'finite':finite,'whole_model_forwards':forwards,'test_evaluations':0};print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and finite and audits['validation_only'] and max(captureerrors)<1e-4 and max(fullerrors)<1e-4
 torch.save({'native_logits':native,'native_heads':heads},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
 for s in summary:
  if s['budget'] in [8,32,64] and s['method']!='random_topk':print(s,flush=True)
if __name__=='__main__':main()

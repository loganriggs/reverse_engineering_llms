"""H759 frozen featurizers across five task families and nuisance edits."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_verified_head_sparse import SignedBSF
from run_bilin18_l5_shared_group_memberships import select_pairs,union_pairs
from run_bilin18_l5_natural_featurizer_baselines import reconstruct
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_diverse_featurizer_transfer'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'panel':ART/'bilin18_fresh_head_tasks_captures.pt','native_results':ART/'bilin18_fresh_head_tasks.json','featurizers':ART/'bilin18_l5_sparse_featurizers_captures.pt','fits':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','memberships':ART/'bilin18_l5_shared_group_memberships_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py')};hashes={k:sha(p) for k,p in paths.items()}
 cfg={'experiment':'H759','hashes':hashes,'design':'Frozen scalar/block/shared-group models all3seeds, all3degree-null rewires per dictionary, fixed/adaptive writePCA8 and3randomTopK8 controls. Reuse H732five-task/two-format panel including known native failures and nuisance changes. Capture L5H3/H5 final-position heads; apply F(donor)-F(receiver) only at final position, two reciprocal task directions, one nuisance and one identity. No retraining or task-based model/seed selection. Full-head reference for every edit.','relative_error_gate':'Native family/format competence pass AND full-head margin-change RMS>=0.25logits; otherwise absolute RMS only. All cells retained.','limitations':'Research-panel reuse, not entirely new tasks/prompts. New application of these frozen L5featurizers. Final-position only, unlike earlier two-position operation tests. Diverse full-head effects, no downstream Q/K clamp. Ratios across small/correlated panels descriptive.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 panel=torch.load(paths['panel'],weights_only=False);prior=json.loads(paths['native_results'].read_text());examples=panel['examples'];native=panel['native_logits'];fits=torch.load(paths['fits'],weights_only=False)['fits'];states=torch.load(paths['featurizers'],weights_only=False)['states'];memberships=torch.load(paths['memberships'],weights_only=False)['memberships'];variants=['full','write_pca','adaptive_write_pca']+[f'{kind}_{seed}' for kind in ['scalar','block','group'] for seed in [7551,7552,7553]]+[f'null_{seed}_{ns}' for seed in [7551,7552,7553] for ns in [1,2,3]]+[f'random_{seed}' for seed in [7541,7542,7543]]
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 heads={};buckets=defaultdict(list);native_errors=[];forwards=0
 for key,e in examples.items():buckets[len(e['ids'])].append(key)
 for length,keys in buckets.items():
  for lo in range(0,len(keys),12):
   bb=keys[lo:lo+12];saved={}
   def capture(m,args):saved['heads']=args[0][:,-1].cpu()
   hook=model.blocks[5].attn.c_proj.register_forward_pre_hook(capture)
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:hook.remove()
   forwards+=len(bb)
   for i,key in enumerate(bb):heads[key]=saved['heads'][i];native_errors.append(float((logits[i].cpu()-native[key]).abs().max()))
 keys=list(heads);recon={};unionerrors=[]
 for h in [3,5]:
  site=f'L5H{h}';fit=fits[site];flat=torch.stack([heads[k][h*128:(h+1)*128] for k in keys]).double()
  for variant in variants:
   if variant=='full':rec=flat
   elif variant in ['write_pca','adaptive_write_pca']:rec=reconstruct(flat,fit,variant,8)
   elif variant.startswith('random_'):rec=reconstruct(flat,fit,'random_topk',8,int(variant.split('_')[1]))
   else:
    parts=variant.split('_');kind,seed=parts[0],int(parts[1]);r=4 if kind=='block' else 1;st=states[site,r,seed];m=SignedBSF(r);m.load_state_dict(st['model']);m.eval();x=((flat-fit['mean'])@fit['write_pca']['encode']/st['input_scale']).float()
    if kind in ['scalar','block']:decoded=m(x)
    else:
     M=memberships[site,seed,'learned_raw' if kind=='group' else 'learned_raw_rewired',None if kind=='group' else int(parts[2])];z=m.encoder(x);pairs=select_pairs(z.square(),M);q=union_pairs(M,pairs);g=torch.zeros(len(x),128).scatter_(1,pairs,1);unionerrors.append(float((q-(g@M>0).float()).abs().max()));decoded=(z*q)@m.decoder+m.bias
    rec=fit['mean']+(decoded.double()*st['input_scale'])@fit['write_pca']['decode']
   for i,key in enumerate(keys):recon[key,h,variant]=rec[i].float()
 print('CAPTURED',forwards,'VARIANTS',len(variants),flush=True);jobs=[]
 for rid in sorted({k[0] for k in examples}):
  for rv,dv,mode in [(0,1,'task'),(1,0,'task'),(0,2,'nuisance'),(0,0,'identity')]:
   for h in [3,5]:
    for variant in variants:jobs.append({'key':(rid,rv),'donor':(rid,dv),'mode':mode,'head':h,'variant':variant})
 buckets=defaultdict(list)
 for j in jobs:buckets[j['head'],len(examples[j['key']]['ids'])].append(j)
 records=[];identity=[];full_vector_errors=[];finite=True
 for (h,length),jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];patch=torch.stack([recon[j['donor'],h,j['variant']]-recon[j['key'],h,j['variant']] for j in bb]).cuda()
   def edit(m,args):
    y=args[0].clone();y[:,-1,h*128:(h+1)*128]+=patch.to(y)
    for i,j in enumerate(bb):
     if j['variant']=='full':full_vector_errors.append(float((y[i,-1,h*128:(h+1)*128].cpu()-heads[j['donor']][h*128:(h+1)*128]).abs().max()))
    return(y,)
   hook=model.blocks[5].attn.c_proj.register_forward_pre_hook(edit)
   try:logits,_=run(model,torch.tensor([examples[j['key']]['ids'] for j in bb],device='cuda'))
   finally:hook.remove()
   forwards+=len(bb);finite=finite and bool(torch.isfinite(logits).all())
   for i,j in enumerate(bb):
    key=j['key'];e=examples[key];p=logits[i].cpu();n=native[key];delta=measure(p,e['groups'])['margin']-measure(n,e['groups'])['margin'];sign=1 if j['donor'][1] in [0,2] else -1;lp=F.log_softmax(p,dim=-1);ln=F.log_softmax(n,dim=-1);records.append({'id':key[0],'receiver':key[1],'donor':j['donor'][1],'head':h,'variant':j['variant'],'mode':j['mode'],'family':e['family'],'format':e['split'],'competent':prior['summary'][e['family']][e['split']]['pass'],'margin_change':delta,'donor_directed_margin_change':sign*delta,'native_to_edit_KL':float((ln.exp()*(ln-lp)).sum())})
    if j['mode']=='identity':identity.append(float((p-n).abs().max()))
 refs={(r['id'],r['receiver'],r['donor'],r['head']):r for r in records if r['variant']=='full'};groups=defaultdict(list)
 for r in records:
  if r['mode']=='identity':continue
  f=refs[r['id'],r['receiver'],r['donor'],r['head']];groups[r['head'],r['family'],r['format'],r['mode'],r['variant'],r['competent']].append((r,f))
 summary=[]
 for (h,fam,fmt,mode,variant,competent),rr in groups.items():
  rms=(sum(f['margin_change']**2 for r,f in rr)/len(rr))**.5;err=(sum((r['margin_change']-f['margin_change'])**2 for r,f in rr)/len(rr))**.5;summary.append({'head':h,'family':fam,'format':fmt,'mode':mode,'variant':variant,'competent':competent,'contexts':len(rr),'full_margin_RMS':rms,'absolute_RMS_error':err,'relative_RMS_error':err/rms if competent and rms>=.25 else None,'mean_full_donor_directed_change':sum(f['donor_directed_margin_change'] for r,f in rr)/len(rr)})
 audits={'hashes_unchanged':{k:sha(p)==hashes[k] for k,p in paths.items()},'native_logit_replay_max_abs':max(native_errors),'identity_logit_error_max_abs':max(identity),'full_donor_head_vector_error_max_abs':max(full_vector_errors),'exact_union_max_abs_error':max(unionerrors),'finite':finite,'whole_model_forwards':forwards,'test_evaluations':0};print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and finite and max(native_errors)<1e-4 and max(identity)<1e-4 and max(full_vector_errors)<1e-4 and max(unionerrors)==0
 torch.save({'native_heads':heads},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'native_competence':prior['summary'],'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
 for s in summary:
  if s['variant']=='full':print(s,flush=True)
if __name__=='__main__':main()

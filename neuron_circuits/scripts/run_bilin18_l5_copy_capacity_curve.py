"""H762 frozen featurizers across five task families and nuisance edits."""
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
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_broad_task_competence import continuation_id
STEM='bilin18_l5_copy_capacity_curve'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'capture':ART/'bilin18_l5_fresh_copy_positions_captures.pt','results':ART/'bilin18_l5_fresh_copy_positions.json','fits':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py')};hashes={k:sha(p) for k,p in paths.items()}
 cfg={'experiment':'H762','hashes':hashes,'budgets':[8,16,32,64,128],'design':'Frozen H754 train-only writePCA/adaptivewritePCA and3random orthogonal write-coordinate transforms, no new fitting. H5 only because H761H3effects were weak. Reuse all32fresh-copy pairs, task/nuisance/identity, query/source/both positions. Same F(donor)-F(receiver) intervention; compare full raw reference. Fixed PCA rank versus adaptive active-coordinate budgets distinguished. K8 and full-rank replay audited.','limitations':'Capacity diagnostic on already-inspected panel, not independent confirmation. No retrained sparse models. Larger budgets need not identify meaningful features. Natural reconstruction and causal-effect approximation distinct. Original reserved test untouched.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 cap=torch.load(paths['capture'],weights_only=False);old=json.loads(paths['results'].read_text());native=cap['native_logits'];examples=cap['examples'];heads=cap['native_heads'];prior={'summary':old['native_competence']};excluded=old['excluded'];fits=torch.load(paths['fits'],weights_only=False)['fits'];specs={'full':None}
 for budget in cfg['budgets']:
  specs[f'write_pca_{budget}']=('write_pca',budget,None);specs[f'adaptive_write_pca_{budget}']=('adaptive_write_pca',budget,None)
  for seed in [7541,7542,7543]:specs[f'random_{seed}_{budget}']=('random_topk',budget,seed)
 variants=list(specs)

 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 forwards=0;keys=list(heads);recon={}
 for h in [5]:
  fit=fits[f'L5H{h}'];flat=torch.stack([heads[k][:,h*128:(h+1)*128] for k in keys]).double().reshape(-1,128)
  for variant,spec in specs.items():
   rec=flat if spec is None else reconstruct(flat,fit,*spec)
   rec=rec.reshape(len(keys),2,128)
   for i,key in enumerate(keys):recon[key,h,variant]=rec[i].float()
 print('CAPTURED',forwards,'VARIANTS',len(variants),flush=True);jobs=[]
 for rid in sorted({k[0] for k in examples}):
  for rv,dv,mode in [(0,1,'task'),(1,0,'task'),(0,2,'nuisance'),(0,0,'identity')]:
   for h in [5]:
    for variant in variants:
     for position in ['query','source','both']:jobs.append({'key':(rid,rv),'donor':(rid,dv),'mode':mode,'head':h,'variant':variant,'position':position})
 buckets=defaultdict(list)
 for j in jobs:buckets[j['head'],len(examples[j['key']]['ids'])].append(j)
 records=[];identity=[];full_vector_errors=[];finite=True
 for (h,length),jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];patch=torch.stack([recon[j['donor'],h,j['variant']]-recon[j['key'],h,j['variant']] for j in bb]).cuda()
   def edit(m,args):
    y=args[0].clone()
    for i,j in enumerate(bb):
     which=[0,1] if j['position']=='both' else [0 if j['position']=='query' else 1]
     for posidx in which:
      pos=length-1 if posidx==0 else examples[j['key']]['source_position'];y[i,pos,h*128:(h+1)*128]+=patch[i,posidx].to(y)
      if j['variant']=='full':full_vector_errors.append(float((y[i,pos,h*128:(h+1)*128].cpu()-heads[j['donor']][posidx,h*128:(h+1)*128]).abs().max()))
    return(y,)
   hook=model.blocks[5].attn.c_proj.register_forward_pre_hook(edit)
   try:logits,_=run(model,torch.tensor([examples[j['key']]['ids'] for j in bb],device='cuda'))
   finally:hook.remove()
   forwards+=len(bb);finite=finite and bool(torch.isfinite(logits).all())
   for i,j in enumerate(bb):
    key=j['key'];e=examples[key];p=logits[i].cpu();n=native[key];delta=measure(p,e['groups'])['margin']-measure(n,e['groups'])['margin'];sign=1 if j['donor'][1] in [0,2] else -1;lp=F.log_softmax(p,dim=-1);ln=F.log_softmax(n,dim=-1);records.append({'id':key[0],'receiver':key[1],'donor':j['donor'][1],'head':h,'variant':j['variant'],'position':j['position'],'mode':j['mode'],'family':e['family'],'format':e['split'],'competent':prior['summary'][e['family']][e['split']]['pass'],'margin_change':delta,'donor_directed_margin_change':sign*delta,'native_to_edit_KL':float((ln.exp()*(ln-lp)).sum())})
    if j['mode']=='identity':identity.append(float((p-n).abs().max()))
 refs={(r['id'],r['receiver'],r['donor'],r['head'],r['position']):r for r in records if r['variant']=='full'};groups=defaultdict(list)
 for r in records:
  if r['mode']=='identity':continue
  f=refs[r['id'],r['receiver'],r['donor'],r['head'],r['position']];groups[r['head'],r['family'],r['format'],r['mode'],r['variant'],r['competent'],r['position']].append((r,f))
 summary=[]
 for (h,fam,fmt,mode,variant,competent,position),rr in groups.items():
  rms=(sum(f['margin_change']**2 for r,f in rr)/len(rr))**.5;err=(sum((r['margin_change']-f['margin_change'])**2 for r,f in rr)/len(rr))**.5;summary.append({'head':h,'family':fam,'format':fmt,'mode':mode,'variant':variant,'position':position,'competent':competent,'contexts':len(rr),'full_margin_RMS':rms,'absolute_RMS_error':err,'relative_RMS_error':err/rms if competent and rms>=.25 else None,'mean_full_donor_directed_change':sum(f['donor_directed_margin_change'] for r,f in rr)/len(rr)})
 oldindex={(r['id'],r['receiver'],r['donor'],r['head'],r['position'],r['variant']):r for r in old['records']};replay=[];fullrank=[]
 for r in records:
  if r['variant']=='full' or r['variant'].endswith('_8'):
   variant=r['variant'] if r['variant']=='full' else r['variant'][:-2];previous=oldindex[r['id'],r['receiver'],r['donor'],r['head'],r['position'],variant];replay.append(abs(r['margin_change']-previous['margin_change']))
  if r['variant'].endswith('_128'):fullrank.append(abs(r['margin_change']-refs[r['id'],r['receiver'],r['donor'],r['head'],r['position']]['margin_change']))
 assert max(replay)<1e-4 and max(fullrank)<1e-4
 audits={'H761_K8_and_full_replay_max_abs':max(replay),'full_rank_margin_replay_max_abs':max(fullrank),'hashes_unchanged':{k:sha(p)==hashes[k] for k,p in paths.items()},'identity_logit_error_max_abs':max(identity),'full_donor_head_vector_error_max_abs':max(full_vector_errors),'finite':finite,'whole_model_forwards':forwards,'test_evaluations':0};print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and finite and max(identity)<1e-4 and max(full_vector_errors)<1e-4
 (ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'native_competence':prior['summary'],'excluded':excluded,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
 for s in summary:
  if s['variant']=='full':print(s,flush=True)
if __name__=='__main__':main()

"""H761 frozen featurizers across five task families and nuisance edits."""
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
STEM='bilin18_l5_fresh_copy_positions'
def make_panel(tok):
 keys=['red','blue','green','yellow','black','white','brown','purple'];values=['hammer','nail','screw','bolt','wheel','blade','brush','rope'];rows=[];examples={};excluded=[]
 for fmt in ['semicolon','newline','colon','equals']:
  def prompt(key,value):
   if fmt=='semicolon':return f'{key} {value}; {key} {value}; {key}'
   if fmt=='newline':return f'{key} {value}\n{key} {value}\n{key}'
   if fmt=='colon':return f'{key}: {value}; {key}: {value}; {key}:'
   return f'{key} = {value}, {key} = {value}, {key} ='
  for i,key in enumerate(keys):
   answers=[values[i],values[(i+3)%8]];texts=[prompt(key,a) for a in answers]+[prompt(keys[(i+1)%8],answers[0])];rid=f'h761-{fmt}-{i}';rows.append({'id':rid,'format':fmt,'prompts':texts,'answers':answers});ee=[]
   for v,text in enumerate(texts):
    gs=[[continuation_id(tok,text,' '+a)] for a in answers];ids=tok.encode(text,add_special_tokens=False)
    if any(g[0] is None for g in gs):break
    target=gs[1 if v==1 else 0][0];matches=[p for p,t in enumerate(ids) if t==target]
    if not matches or matches[-1]==len(ids)-1:break
    ee.append({'ids':ids,'groups':gs,'source_position':matches[-1],'family':'sequence_copy','split':fmt})
   if len(ee)!=3:excluded.append({'id':rid,'reason':'single-token answer or unambiguous source-token position'});continue
   for v,e in enumerate(ee):examples[rid,v]=e
 return rows,examples,excluded
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();manifest=ART/f'{STEM}_manifest.json';tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);rows,examples,excluded=make_panel(tok);manifest.write_text(json.dumps(rows,indent=2)+'\n');paths={'manifest':manifest,'featurizers':ART/'bilin18_l5_sparse_featurizers_captures.pt','fits':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','memberships':ART/'bilin18_l5_shared_group_memberships_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py')};hashes={k:sha(p) for k,p in paths.items()}
 cfg={'experiment':'H761','hashes':hashes,'design':'Prospective fresh copying/nuisance panel locked before outputs:8 color keys,8 tool/object values,4 formats,32 pairs and3variants. All H755/H757 fits and controls frozen, all seeds retained. Compare F(donor)-F(receiver) edits at final query, last copied-value source, and both. Two reciprocal task directions, nuisance key-rename and identity. Source position found from the intended answer token in each prompt; token-validity failures retained in exclusions, no outcome filtering.','competence_gate':'Per format >=4validpairs, >=.75bothendpoint choice, >=.5 intended full-vocabulary top1. Nuisance correctness reported separately. Relative effect error only if competence passes and full effect RMS>=.25logits.','limitations':'New panel relative to H732/H747; not asserted absent from pretraining. Handwritten limited task family. Source and final edits change different computations. No downstream node clamps, no task fitting.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 native={};prior={'summary':{'sequence_copy':{}}};fits=torch.load(paths['fits'],weights_only=False)['fits'];states=torch.load(paths['featurizers'],weights_only=False)['states'];memberships=torch.load(paths['memberships'],weights_only=False)['memberships'];variants=['full','write_pca','adaptive_write_pca']+[f'{kind}_{seed}' for kind in ['scalar','block','group'] for seed in [7551,7552,7553]]+[f'null_{seed}_{ns}' for seed in [7551,7552,7553] for ns in [1,2,3]]+[f'random_{seed}' for seed in [7541,7542,7543]]
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 heads={};buckets=defaultdict(list);native_errors=[];forwards=0
 for key,e in examples.items():buckets[len(e['ids'])].append(key)
 for length,keys in buckets.items():
  for lo in range(0,len(keys),12):
   bb=keys[lo:lo+12];saved={}
   def capture(m,args):saved['heads']=torch.stack([args[0][i,[len(examples[k]['ids'])-1,examples[k]['source_position']]] for i,k in enumerate(bb)]).cpu()
   hook=model.blocks[5].attn.c_proj.register_forward_pre_hook(capture)
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:hook.remove()
   forwards+=len(bb)
   for i,key in enumerate(bb):heads[key]=saved['heads'][i];native[key]=logits[i].cpu()
 for fmt in ['semicolon','newline','colon','equals']:
  ids0=sorted({k[0] for k,e in examples.items() if e['split']==fmt});mm={(rid,v):measure(native[rid,v],examples[rid,v]['groups']) for rid in ids0 for v in [0,1,2]};both=sum(mm[rid,0]['margin']>0 and mm[rid,1]['margin']<0 for rid in ids0)/max(len(ids0),1);top=sum(mm[rid,v]['top1'][v] for rid in ids0 for v in [0,1])/max(2*len(ids0),1);prior['summary']['sequence_copy'][fmt]={'pairs':len(ids0),'both_endpoint_choice':both,'full_vocab_top1':top,'nuisance_choice':sum(mm[rid,2]['margin']>0 for rid in ids0)/max(len(ids0),1),'pass':len(ids0)>=4 and both>=.75 and top>=.5}
 print('COMPETENCE',prior['summary'],flush=True)
 keys=list(heads);recon={};unionerrors=[]
 for h in [3,5]:
  site=f'L5H{h}';fit=fits[site];flat=torch.stack([heads[k][:,h*128:(h+1)*128] for k in keys]).double().reshape(-1,128)
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
   rec=rec.reshape(len(keys),2,128)
   for i,key in enumerate(keys):recon[key,h,variant]=rec[i].float()
 print('CAPTURED',forwards,'VARIANTS',len(variants),flush=True);jobs=[]
 for rid in sorted({k[0] for k in examples}):
  for rv,dv,mode in [(0,1,'task'),(1,0,'task'),(0,2,'nuisance'),(0,0,'identity')]:
   for h in [3,5]:
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
 audits={'hashes_unchanged':{k:sha(p)==hashes[k] for k,p in paths.items()},'identity_logit_error_max_abs':max(identity),'full_donor_head_vector_error_max_abs':max(full_vector_errors),'exact_union_max_abs_error':max(unionerrors),'finite':finite,'whole_model_forwards':forwards,'test_evaluations':0};print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and finite and max(identity)<1e-4 and max(full_vector_errors)<1e-4 and max(unionerrors)==0
 torch.save({'native_heads':heads,'native_logits':native,'examples':examples},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'native_competence':prior['summary'],'excluded':excluded,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
 for s in summary:
  if s['variant']=='full':print(s,flush=True)
if __name__=='__main__':main()

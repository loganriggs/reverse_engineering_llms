"""H777 frozen copying contrasts across crossed lexical cohorts."""
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
STEM='bilin18_l5_copy_lexical_transfer'
def make_lexical_panel(tok):
 keys={'old':['red','blue','green','yellow','black','white','brown','purple'],'new':['cat','dog','bird','fish','horse','bear','wolf','deer']};values={'old':['hammer','nail','screw','bolt','wheel','blade','brush','rope'],'new':['apple','banana','orange','lemon','peach','pear','grape','mango']};rows=[];examples={};excluded=[];value_tokens={'old':set(),'new':set()}
 for kc,kk in keys.items():
  for vc,vv in values.items():
   for fmt in ['semicolon','newline','colon','equals']:
    def prompt(key,value):
     if fmt=='semicolon':return f'{key} {value}; {key} {value}; {key}'
     if fmt=='newline':return f'{key} {value}\n{key} {value}\n{key}'
     if fmt=='colon':return f'{key}: {value}; {key}: {value}; {key}:'
     return f'{key} = {value}, {key} = {value}, {key} ='
    for i,key in enumerate(kk):
     answers=[vv[i],vv[(i+3)%8]];texts=[prompt(key,a) for a in answers]+[prompt(kk[(i+1)%8],answers[0])];rid=f'h777-{kc}-{vc}-{fmt}-{i}';row={'id':rid,'keys':kc,'values':vc,'format':fmt,'item':i,'prompts':texts,'answers':answers};rows.append(row);ee=[]
     for v,text in enumerate(texts):
      gs=[[continuation_id(tok,text,' '+a)] for a in answers];ids=tok.encode(text,add_special_tokens=False)
      if any(g[0] is None for g in gs):break
      target=gs[1 if v==1 else 0][0];matches=[p for p,t in enumerate(ids) if t==target]
      if not matches or matches[-1]==len(ids)-1:break
      value_tokens[vc].update(g[0] for g in gs);ee.append({'ids':ids,'groups':gs,'source_position':matches[-1],'family':'sequence_copy','split':f'{kc}_keys_{vc}_values|{fmt}','lexical_cohort':f'{kc}_keys_{vc}_values','format':fmt,'item':i})
     if len(ee)!=3:excluded.append({'id':rid,'reason':'single-token answer/source-position validity'});continue
     for v,e in enumerate(ee):examples[rid,v]=e
 assert value_tokens['old'].isdisjoint(value_tokens['new'])
 return rows,examples,excluded,{k:sorted(v) for k,v in value_tokens.items()}
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'frozen':ART/'bilin18_l5_copy_contrast_subspaces_fits.pt','results':ART/'bilin18_l5_copy_contrast_subspaces.json','fits':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);rows,examples,excluded,value_tokens=make_lexical_panel(tok);(ART/f'{STEM}_manifest.json').write_text(json.dumps(rows,indent=2)+'\n');cfg={'experiment':'H777','hashes':hashes,'design':'Cross old/newkeys with old/newvalues over4fixedformats and8items,128plannedpairs. Values new fruits, old tools; keys new animals, old colors. All H776rank8bases frozen, no fitting/selection. Query/source/both task/nuisance/identity edits; native competence gated per lexicalcohort/format. New value continuationtokenids verified disjoint from old values. Old/old is replay reference, not novel evaluation.','limitations':'One category shift per lexical axis; no claim unseen in pretraining. Oldvalues mean original8-word panel, not all fittedtrainingpairs. Prospective newlexicalcohorts but familiar formats. Small correlated panels.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');old=json.loads(paths['results'].read_text());fit=torch.load(paths['fits'],weights_only=False)['fits']['L5H5'];E=fit['write_pca']['encode'];B=fit['write_pca']['decode'];frozen=torch.load(paths['frozen'],weights_only=False);bases=frozen['bases'];spectra=frozen['spectra'];variants=['full']+list(bases);orth=max(float((P.T@P-torch.eye(8)).abs().max()) for P in bases.values());fithash=sha(paths['frozen']);model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native={};heads={};capture_forwards=0;buckets=defaultdict(list)
 for key,e in examples.items():buckets[len(e['ids'])].append(key)
 for length,kk in buckets.items():
  for lo in range(0,len(kk),12):
   bb=kk[lo:lo+12];saved={}
   def capture(m,args):saved['heads']=torch.stack([args[0][i,[length-1,examples[k]['source_position']]] for i,k in enumerate(bb)]).cpu()
   hook=model.blocks[5].attn.c_proj.register_forward_pre_hook(capture)
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:hook.remove()
   capture_forwards+=len(bb)
   for i,key in enumerate(bb):heads[key]=saved['heads'][i];native[key]=logits[i].cpu()
 prior={'summary':{'sequence_copy':{}}}
 for split in sorted({e['split'] for e in examples.values()}):
  rids=sorted({k[0] for k,e in examples.items() if e['split']==split});mm={(rid,v):measure(native[rid,v],examples[rid,v]['groups']) for rid in rids for v in [0,1,2]};both=sum(mm[rid,0]['margin']>0 and mm[rid,1]['margin']<0 for rid in rids)/len(rids);top=sum(mm[rid,v]['top1'][v] for rid in rids for v in [0,1])/(2*len(rids));prior['summary']['sequence_copy'][split]={'pairs':len(rids),'both_endpoint_choice':both,'full_vocab_top1':top,'nuisance_choice':sum(mm[rid,2]['margin']>0 for rid in rids)/len(rids),'pass':len(rids)>=4 and both>=.75 and top>=.5}
 print('COMPETENCE',prior,'excluded',excluded,flush=True)
 def delta(rid,v,pos):return (heads[rid,v][pos,640:768].double()-heads[rid,0][pos,640:768].double())@E
 forwards=capture_forwards;keys=list(heads);recon={};flat=torch.stack([heads[k][:,640:768] for k in keys]).double().reshape(-1,128)
 for variant in variants:
  rec=flat if variant=='full' else (flat-fit['mean'])@E@bases[variant]@bases[variant].T@B+fit['mean'];rec=rec.reshape(len(keys),2,128)
  for i,key in enumerate(keys):recon[key,5,variant]=rec[i].float()
 def cohort(rid):return examples[rid,0]['lexical_cohort']
 coverage=[]
 for rid in sorted({k[0] for k in examples}):
  for v,mode in [(1,'task'),(2,'nuisance')]:
   for pos in [0,1]:
    D=delta(rid,v,pos)
    for name,P in bases.items():coverage.append({'id':rid,'cohort':cohort(rid),'mode':mode,'position':['query','source'][pos],'variant':name,'full_write_squared_norm':float(D.square().sum()),'residual_write_squared_norm':float((D-D@P@P.T).square().sum())})
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
 oldindex={(r['id'],r['receiver'],r['donor'],r['head'],r['position'],r['variant']):r for r in old['records']};replay=[]
 for r in records:
  e=examples[r['id'],0];r['cohort']=e['lexical_cohort'];r['template']=e['format'];r['item']=e['item']
  if r['cohort']=='old_keys_old_values':
   oldrid=f"h761-{e['format']}-{e['item']}";replay.append(abs(r['margin_change']-oldindex[oldrid,r['receiver'],r['donor'],r['head'],r['position'],r['variant']]['margin_change']))
 audits={'H776_oldold_margin_replay_max_abs':max(replay),'fit_hash_unchanged':sha(paths['frozen'])==fithash,'basis_orthogonality_max_abs':orth,'new_value_tokens_disjoint':set(value_tokens['old']).isdisjoint(value_tokens['new']),'valid_pairs':len(examples)//3,'excluded_pairs':len(excluded),'hashes_unchanged':{k:sha(p)==hashes[k] for k,p in paths.items()},'identity_logit_error_max_abs':max(identity),'full_donor_head_vector_error_max_abs':max(full_vector_errors),'finite':finite,'whole_model_forwards':forwards,'test_evaluations':0};print(audits,flush=True);assert audits['fit_hash_unchanged'] and audits['new_value_tokens_disjoint'] and max(replay)<1e-4 and all(audits['hashes_unchanged'].values()) and finite and max(identity)<1e-4 and max(full_vector_errors)<1e-4
 torch.save({'native_heads':heads,'native_logits':native,'examples':examples},ART/f'{STEM}_captures.pt')
 (ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'native_competence':prior['summary'],'excluded':excluded,'records':records,'summary':summary,'vector_coverage':coverage,'spectra':spectra,'value_tokens':value_tokens,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
 for s in summary:
  if s['variant']=='full':print(s,flush=True)
if __name__=='__main__':main()

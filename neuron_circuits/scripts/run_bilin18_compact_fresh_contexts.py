"""H952 frozen relay bases on new lexical pairs and distance variants."""
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure,groups,forward
from run_bilin18_broad_task_competence import continuation_id
from run_bilin18_relay_contrast_subspaces import run
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_compact_fresh_contexts'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'fits':ART/'bilin18_relay_contrast_subspaces_fits.pt','frozen_students':ART/'bilin18_product_complexity_training_fits.pt','runner':Path(__file__),'patch_runner':ROOT/'scripts/run_bilin18_relay_contrast_subspaces.py','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H952','hashes':hashes,'design':'Prospective96 pairs: eight new source pairs/task, two formats, three fixed gaps. No replacement based on model outputs. Basis/layer/target position frozen from H848. Native gate per family/format/gap:>=4 valid pairs, >=.75 both-endpoint binary, >=.5 fullvocab intendedgroup. All valid pairs intervened including gate failures, reported separately.','fresh_scope':'New full prompt/lexical-pair panel relative to H840-H848; some vocabulary/template elements reused and no claim of pretraining or entire-research disjointness. No fitting.','distance':'Agreement gaps empty/right/over there before near or beside. Sentiment empty/indeed/overall before period. Gaps change wording as well as token distance, not a pure distance manipulation.','arms':'Full donor preMLP4 relay; own rank1/2/4/8 projected contrast; rank1 complement; othertask rank1; matchednorm random direction plus/minus using H848 random0. Identity and patch-input checks.','interpretation':'Task contrast compression, not unsupervised manifold recovery. Donor contrast supplies coefficients. Site/coordinate system must stay fixed; no fallback relay selection on failed gap variants.'};cfg.update(design='Prospective64 pairs: eight source pairs per task, four fixed context conditions. Frozen H949 fits and H848 relay bases. No output-dependent replacements. Native competence gate unchanged: >=4 valid pairs, >=.75 both endpoints binary, >=.5 intended fullvocab. Retain all valid pairs including failed cells.',fresh_scope='Full prompts checked disjoint from H849 and original discovery source; new source list, not entire-research/pretraining disjointness.',distance='Context composition intervention, not controlled distance: relative clauses/opposite-number attractors for agreement; negation/opposing opening for sentiment; baseline and alternate context controls.',arms='Full donor preMLP4 relay; own rank1; complement; other-task rank1. Site fixed, same relay token and absolute position required.');(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');pairs=[]
 nouns=[('pilot','pilots'),('chef','chefs'),('farmer','farmers'),('dancer','dancers'),('singer','singers'),('soldier','soldiers'),('lawyer','lawyers'),('painter','painters')]
 adjs=[('excellent','awful'),('wonderful','terrible'),('superb','horrible'),('delightful','miserable'),('brilliant','boring'),('enjoyable','tedious'),('satisfying','disappointing'),('impressive','disgusting')]
 for f,source in [('agreement',nouns),('sentiment',adjs)]:
  for fmt in ['baseline','composition','distractor','alternate']:
   for k,values in enumerate(source):
    if f=='agreement':
     suffixes={'baseline':['',''],'composition':[' that the guards admire',' that the guard admires'],'distractor':[' with the guards',' with the guard'],'alternate':[' beside the old building',' beside the old building']}[fmt]
     texts=[f'The {v}{suffix} near the window' for v,suffix in zip(values,suffixes)];answers=['is','are'];offset=-3;target=' near'
    else:
     if fmt=='baseline':texts=[f'Review: The performance was {v}. Sentiment:' for v in values]
     elif fmt=='composition':texts=[f'Review: The performance was not {v}. Sentiment:' for v in values[::-1]]
     elif fmt=='distractor':texts=[f'Review: Although the opening was {opposite}, the performance overall was {v}. Sentiment:' for v,opposite in zip(values,values[::-1])]
     else:texts=[f'Review: I found the performance {v}. Sentiment:' for v in values]
     answers=['positive','negative'];offset=-4;target='.'
    pairs.append({'id':f'{f}-{fmt}-{k}','family':f,'format':fmt,'gap':0,'source_pair':k,'prompts':texts,'answers':answers,'offset':offset,'target':target})
 (ART/f'{S}_manifest.json').write_text(json.dumps(pairs,indent=2)+'\n');manifest_hash=sha(ART/f'{S}_manifest.json');oldtexts=set();
 for source in ['bilin18_relay_fresh_distance_captures.pt','bilin18_candidate_task_screen_captures.pt']:
  oldtexts.update(e['text'] for e in torch.load(ART/source,weights_only=False)['examples'])
 assert all(t not in oldtexts for p in pairs for t in p['prompts']), 'Exact prior prompt overlap'
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);ex=[];valid=[];excluded=[]
 for p in pairs:
  ee=[]
  for v,text in enumerate(p['prompts']):
   ids=tok.encode(text,add_special_tokens=False);gg=[[continuation_id(tok,text,c) for c in g] for g in groups(p['family'],p['answers'])];gg=[[t for t in g if t is not None] for g in gg];pos=len(ids)+p['offset']
   if any(not g for g in gg) or set(gg[0])&set(gg[1]) or tok.decode([ids[pos]])!=p['target']:break
   ee.append({'id':p['id'],'variant':v,'text':text,'ids':ids,'groups':gg,'position':pos})
  if len(ee)!=2:excluded.append({'id':p['id'],'reason':'answer or target tokenization'});continue
  if len(ee[0]['ids'])!=len(ee[1]['ids']):excluded.append({'id':p['id'],'reason':'unequal token lengths'});continue
  valid.append(p);ex.extend(ee)
 ix={(e['id'],e['variant']):i for i,e in enumerate(ex)};model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native={};caps={};actual=0;ident=0.;positionaudit=True
 for length in sorted({len(e['ids']) for e in ex}):
  inds=[i for i,e in enumerate(ex) if len(e['ids'])==length]
  for st in range(0,len(inds),24):
   batch=inds[st:st+24];ids=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');positions=torch.tensor([ex[i]['position'] for i in batch],device='cuda');x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
   for l,block in enumerate(model.blocks):
    raw=block.lambdas[0]*x+block.lambdas[1]*x0;att,bus=block.attn(F.rms_norm(raw,(1152,)),bus);pre=raw+att
    if l==4:cap=pre[torch.arange(len(batch),device='cuda'),positions].cpu()
    x=pre+block.mlp(F.rms_norm(pre,(1152,)))
   y=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30);yy=run(model,ids,cap.cuda(),positions);actual+=2*len(batch);ident=max(ident,float((yy-y).abs().max()))
   for k,i in enumerate(batch):native[i]=y[k].double().cpu();caps[i]=cap[k]
 competence=[]
 for f in ['agreement','sentiment']:
  for fmt in ['baseline','composition','distractor','alternate']:
   for gap in [0]:
    pp=[p for p in valid if (p['family'],p['format'],p['gap'])==(f,fmt,gap)];both=[];tops=[]
    for p in pp:
     m=[measure(native[ix[p['id'],v]],ex[ix[p['id'],v]]['groups']) for v in [0,1]];both.append(m[0]['margin']>0 and m[1]['margin']<0);tops.extend([m[v]['top1'][v] for v in [0,1]])
    c={'family':f,'format':fmt,'gap':gap,'pairs':len(pp),'both_correct':float(np.mean(both)) if both else 0,'full_vocab':float(np.mean(tops)) if tops else 0};c['pass']=c['pairs']>=4 and c['both_correct']>=.75 and c['full_vocab']>=.5;competence.append(c)
 bases=torch.load(paths['fits'],weights_only=False);jobs=[];full={};records=[]
 for p in valid:
  for rec,don in [(0,1),(1,0)]:
   i,j=ix[p['id'],rec],ix[p['id'],don];assert ex[i]['position']==ex[j]['position'] and ex[i]['ids'][ex[i]['position']]==ex[j]['ids'][ex[j]['position']];delta=(caps[j]-caps[i]).double();B=bases['bases'][p['family']];other=bases['bases']['sentiment' if p['family']=='agreement' else 'agreement'];proj=delta@B[:,:1]@B[:,:1].T;arms=[('full',0,delta),('complement',1,delta-proj),('other',1,delta@other[:,:1]@other[:,:1].T)]+[('own',rank,delta@B[:,:rank]@B[:,:rank].T) for rank in [1]]
   for kind,rank,edit in arms:jobs.append((p,rec,don,i,j,kind,rank,edit))
 for length in sorted({len(e['ids']) for e in ex}):
  jj=[j for j in jobs if len(ex[j[3]]['ids'])==length]
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];ids=torch.tensor([ex[i]['ids'] for p,rec,don,i,j,kind,rank,edit in batch],device='cuda');positions=torch.tensor([ex[i]['position'] for p,rec,don,i,j,kind,rank,edit in batch],device='cuda');replacement=torch.stack([caps[i].double()+edit for p,rec,don,i,j,kind,rank,edit in batch]).cuda();y=run(model,ids,replacement,positions);actual+=len(batch)
   for k,(p,rec,don,i,j,kind,rank,edit) in enumerate(batch):
    b=measure(native[i],ex[i]['groups'])['margin'];v=measure(y[k],ex[i]['groups'])['margin'];dm=measure(native[j],ex[j]['groups'])['margin'];sgn=1 if rec==1 else -1;records.append({key:p[key] for key in ['id','family','format','gap','source_pair']}|{'recipient':rec,'donor':don,'kind':kind,'rank':rank,'donor_directed_delta':sgn*(v-b),'native_answer_gap':sgn*(dm-b)})
  print('DONE',length,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'manifest_unchanged':manifest_hash==sha(ART/f'{S}_manifest.json'),'identity_logits_max_abs':ident,'same_token_same_position':positionaudit,'exact_prior_prompt_overlap':0,'frozen_student_fits_unchanged':sha(paths['frozen_students'])==hashes['frozen_students'],'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['donor_directed_delta']) for r in records)};assert audits['hashes_unchanged'] and audits['manifest_unchanged'] and ident<1e-4 and audits['finite'];torch.save({'native':native,'premlp_relay':caps,'examples':ex},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'manifest_sha256':manifest_hash,'competence':competence,'excluded':excluded,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'competence':competence,'excluded':excluded,'audits':audits}),flush=True)
if __name__=='__main__':main()

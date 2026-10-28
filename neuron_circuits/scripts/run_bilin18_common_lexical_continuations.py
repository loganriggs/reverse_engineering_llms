"""H869: common lexical sources across copy, label binding and capital queries."""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_broad_task_competence import continuation_id,final_logits
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_common_lexical_continuations'
PAIRS=[(('France','Paris'),('Germany','Berlin')),(('Italy','Rome'),('Spain','Madrid')),(('Japan','Tokyo'),('China','Beijing')),(('Canada','Ottawa'),('Australia','Canberra')),(('Norway','Oslo'),('Sweden','Stockholm')),(('Denmark','Copenhagen'),('Finland','Helsinki')),(('Austria','Vienna'),('Hungary','Budapest')),(('Poland','Warsaw'),('Portugal','Lisbon')),(('Greece','Athens'),('Turkey','Ankara')),(('Egypt','Cairo'),('Kenya','Nairobi')),(('Peru','Lima'),('Chile','Santiago')),(('Ireland','Dublin'),('Belgium','Brussels')),(('Switzerland','Bern'),('Netherlands','Amsterdam')),(('Mexico','Mexico City'),('Brazil','Brasilia')),(('Russia','Moscow'),('Ukraine','Kyiv')),(('Thailand','Bangkok'),('Vietnam','Hanoi'))]
TARGETS=[(13,0),(14,4),(8,3)]
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','prior':ART/'bilin18_value_reader_erasure.json'};hashes={k:sha(v) for k,v in paths.items()};cfg={'experiment':'H869','hashes':hashes,'pairs':PAIRS,'targets':TARGETS,'design':'Adaptive afterH868 failure: same16country/capital pairs andsplit,2 predeclared continuation formats,3families,2 answer endpoints. Copy repetition,diplomatic IOI-like binding,direct capitalquery. Common lexicalcontrasts but different sourcepositions/counts; no exactprefix claim. Only sourcecountry tokens change within paired taskendpoints.','gate':'Perfamily/format/query/split at least4 aligned lexicallyeligible pairs, both endpoints binarycorrect>=.75, pooled fullvocabtop1>=.5. Common lexical eligibility requires bothcountry andcapital single-token answers for allqueryvariants; record all exclusions.','scope':'New continuation combinations after failedinstructionpanel; vocabulary familiar and design adaptive. Country-as-agent binding is a controlled diplomatic template, not broad natural language. Freeze targets and nativegate before outcomes; failedcells retained. Capture routing/mixedvalues only; no causal claims.','next':'If competent cells exist, compare same sourcevalue contrasts under differentquery tasks at frozen readers; preserve failedcells as negative controls. No reservedtest.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples=[];excluded=[]
 for pair,(a,b) in enumerate(PAIRS):
  rows=[];errors=[];split='discovery' if pair<8 else 'transfer'
  source_ids=[tok.encode(' '+v[0],add_special_tokens=False) for v in [a,b]]
  if any(len(v)!=1 for v in source_ids):excluded.append({'pair':pair,'reasons':['source_not_single']});continue
  for fmt in ['A','B']:
   for fam in ['copy','binding','capital']:
    rr=[];answers=[v[1 if fam=='capital' else 0] for v in [a,b]]
    for order in [0,1]:
     country=[a,b][order][0];giver=[a,b][1-order][0]
     if fam=='copy':text=f'country {country}; country {country}; country' if fmt=='A' else f'nation {country}, nation {country}, nation'
     elif fam=='binding':text=f'Yesterday, {a[0]} and {b[0]} met. {giver} sent a letter to' if fmt=='A' else f'Earlier, {a[0]} and {b[0]} met. {giver} spoke to'
     else:text=f'The capital of {country} is' if fmt=='A' else f'{country} has its capital in'
     ids=tok.encode(text,add_special_tokens=False);answerids=[continuation_id(tok,text,' '+v) for v in answers]
     if any(x is None for x in answerids):errors.append('answer_not_single:'+','.join(answers));continue
     rr.append(dict(pair=pair,split=split,format=fmt,query=0,family=fam,order=order,expected=order,text=text,ids=ids,groups=[[x] for x in answerids],id=f'{pair}-{fmt}-{fam}'))
    if len(rr)!=2:continue
    if len(rr[0]['ids'])!=len(rr[1]['ids']):errors.append('unequal_endpoint_length');continue
    changed=[i for i,(u,v) in enumerate(zip(rr[0]['ids'],rr[1]['ids'])) if u!=v]
    if not changed:errors.append('no_source_change');continue
    # Directcapital formatB can have a sentence-initial token without leading space.
    legal=set(source_ids[0]+source_ids[1]+tok.encode(a[0],add_special_tokens=False)+tok.encode(b[0],add_special_tokens=False))
    if any(rr[v]['ids'][p] not in legal for p in changed for v in [0,1]):errors.append('non_source_change');continue
    for r in rr:r['source_positions']=changed;r['source_words']=[tok.decode([r['ids'][p]]) for p in changed]
    rows.extend(rr)
  if errors or len(rows)!=12:excluded.append({'pair':pair,'words':[a,b],'reasons':sorted(set(errors))});continue
  examples.extend(rows)
 assert examples;(ART/f'{S}_manifest.json').write_text(json.dumps({'examples':examples,'excluded':excluded},indent=2)+'\n');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 stash={};hooks=[];originals={}
 for l,h in TARGETS:
  att=model.blocks[l].attn;original=att.squared_attention;originals[l]=original
  def capture(q,k,v,q2,k2,layer=l,head=h,fn=original):
   A=(torch.einsum('bhd,bthd->bht',q[:,-1],k)/128)*(torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/128);stash[layer,'routing']=A[:,head].cpu();stash[layer,'values']=v[:,:,head].cpu();return fn(q,k,v,q2,k2)
  att.squared_attention=capture
  def hook(mod,args,layer=l,head=h):stash[layer,'output']=args[0][:,-1,head*128:(head+1)*128].cpu()
  hooks.append(att.c_proj.register_forward_pre_hook(hook))
 buckets=defaultdict(list)
 for i,e in enumerate(examples):buckets[len(e['ids'])].append(i)
 native={};measurements={};captures={};buscaps={};actual=0;audit=0.;reconstruction=0.;static={};staticerr=0.
 for length,ii in sorted(buckets.items()):
  for st in range(0,len(ii),12):
   ix=ii[st:st+12];ids=torch.tensor([examples[i]['ids'] for i in ix],device='cuda');x0=torch.nn.functional.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
   for l,block in enumerate(model.blocks):
    x,bus=block(x,bus,x0)
    if l==0:captured_bus=bus.cpu()
   y=30*torch.tanh(model.lm_head(torch.nn.functional.rms_norm(x,(1152,))[:,-1]).float()/30);actual+=len(ix)
   if not native:
    snapshot={k:v.clone() for k,v in stash.items()};yy=final_logits(model,ids);actual+=len(ix);audit=float((yy-y).abs().max());stash=snapshot
   for k,i in enumerate(ix):
    native[i]=y[k].double().cpu();measurements[i]=measure(native[i],examples[i]['groups']);measurements[i]['predicted_token']=tok.decode([measurements[i]['top_token']]);buscaps[i]=captured_bus[k];captures[i]={}
    for l,h in TARGETS:
     A=stash[l,'routing'][k];V=stash[l,'values'][k];H=stash[l,'output'][k];pred=(A[:,None]*V).sum(0);reconstruction=max(reconstruction,float((pred-H).norm()/H.norm().clamp_min(1e-12)));captures[i][l,h]={'routing':A,'mixed_values':V,'head_output':H}
     for p in examples[i]['source_positions']:
      key=(int(ids[k,p]),h);val=captured_bus[k,p,h]
      if key in static:staticerr=max(staticerr,float((val-static[key]).abs().max()))
      else:static[key]=val.clone()
  print('DONE',length,flush=True)
 for hook in hooks:hook.remove()
 for l,fn in originals.items():model.blocks[l].attn.squared_attention=fn
 cells=[]
 for fam in ['copy','binding','capital']:
  for fmt in ['A','B']:
   for query in [0]:
    for split in ['discovery','transfer']:
     ix=[i for i,e in enumerate(examples) if (e['family'],e['format'],e['query'],e['split'])==(fam,fmt,query,split)];by=defaultdict(list)
     for i in ix:by[examples[i]['pair']].append(i)
     both=[all((measurements[i]['margin']>0)==(examples[i]['expected']==0) for i in inds) for inds in by.values()];top=[measurements[i]['top1'][examples[i]['expected']] for i in ix];c=dict(family=fam,format=fmt,query=query,split=split,pairs=len(by),both_correct=float(np.mean(both)) if both else 0.,full_vocab=float(np.mean(top)) if top else 0.);c['pass']=c['pairs']>=4 and c['both_correct']>=.75 and c['full_vocab']>=.5;cells.append(c)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_forward_max_abs':audit,'signed_AV_relative_error':reconstruction,'source_bus_same_token_max_abs':staticerr,'eligible_lexical_pairs':len(examples)//12,'source_positions_aligned_within_task_pairs':True,'only_source_tokens_change_with_order':True,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(torch.isfinite(v).all().item() for v in native.values())};assert audits['hashes_unchanged'] and audit<1e-4 and reconstruction<1e-4 and staticerr<1e-4 and audits['finite'];torch.save({'examples':examples,'native':native,'captures':captures,'bus':buscaps,'static_bus':static},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'competence':cells,'excluded':excluded,'measurements':measurements,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'competence':cells,'audits':audits}),flush=True)
if __name__=='__main__':main()

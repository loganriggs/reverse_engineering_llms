"""H868: common lexical sources across copy, label binding and capital queries."""
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
S='bilin18_common_source_panel'
PAIRS=[(('France','Paris'),('Germany','Berlin')),(('Italy','Rome'),('Spain','Madrid')),(('Japan','Tokyo'),('China','Beijing')),(('Canada','Ottawa'),('Australia','Canberra')),(('Norway','Oslo'),('Sweden','Stockholm')),(('Denmark','Copenhagen'),('Finland','Helsinki')),(('Austria','Vienna'),('Hungary','Budapest')),(('Poland','Warsaw'),('Portugal','Lisbon')),(('Greece','Athens'),('Turkey','Ankara')),(('Egypt','Cairo'),('Kenya','Nairobi')),(('Peru','Lima'),('Chile','Santiago')),(('Ireland','Dublin'),('Belgium','Brussels')),(('Switzerland','Bern'),('Netherlands','Amsterdam')),(('Mexico','Mexico City'),('Brazil','Brasilia')),(('Russia','Moscow'),('Ukraine','Kyiv')),(('Thailand','Bangkok'),('Vietnam','Hanoi'))]
TARGETS=[(13,0),(14,4),(8,3)]
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','prior':ART/'bilin18_value_reader_erasure.json'};hashes={k:sha(v) for k,v in paths.items()};cfg={'experiment':'H868','hashes':hashes,'pairs':PAIRS,'targets':TARGETS,'design':'16 predeclared country/capital pairs, first8 discovery last8 transfer;2 common-source formats,2 queried slots,3 queryfamilies,2 sourceorders. Exact same sourceprefix across taskqueries, countrytokens exchanged at2source positions. Native competence screen before causalinterpretation, no taskoutcome filtering of individualpairs.','gate':'Perfamily/format/query/split at least4 aligned lexicallyeligible pairs, both endpoints binarycorrect>=.75, pooled fullvocabtop1>=.5. Common lexical eligibility requires bothcountry andcapital single-token answers for allqueryvariants; record all exclusions.','scope':'New controlled combinations, familiar vocabulary may have prior research exposure. Copy here means ordinal source retrieval; binding means label retrieval; not diverse natural tasks. Capture frozen3readers routing/mixedvalues without edits; no causal claim from correlations.','next':'If competent cells exist, compare same sourcevalue contrasts under differentquery tasks at frozen readers; preserve failedcells as negative controls. No reservedtest.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples=[];excluded=[]
 for pair,(a,b) in enumerate(PAIRS):
  rows=[];errors=[];split='discovery' if pair<8 else 'transfer'
  for fmt in ['colon','equals']:
   for query in [0,1]:
    label=['A','B'][query];ordinal=['first','second'][query]
    suffix={'copy':f'Repeat the {ordinal} country:','binding':f'The country labeled {label} is','capital':f'The capital of the country labeled {label} is'}
    for fam,question in suffix.items():
     answers=[a[1 if fam=='capital' else 0],b[1 if fam=='capital' else 0]]
     for order in [0,1]:
      source=[a[0],b[0]] if order==0 else [b[0],a[0]];prefix=f'A: {source[0]}. B: {source[1]}.\n' if fmt=='colon' else f'A = {source[0]}; B = {source[1]}.\n';text=prefix+question;ids=tok.encode(text,add_special_tokens=False);answerids=[continuation_id(tok,text,' '+x) for x in answers];positions=[]
      for word in source:
       enc=tok.encode(' '+word,add_special_tokens=False)
       if len(enc)!=1:errors.append(f'source_not_single:{word}');continue
       occ=[i for i,t in enumerate(ids) if t==enc[0]]
       if len(occ)!=1:errors.append(f'source_occurrence:{word}');continue
       positions.append(occ[0])
      if any(x is None for x in answerids):errors.append('answer_not_single:'+','.join(answers))
      if len(positions)!=2 or any(x is None for x in answerids):continue
      assert len(set(answerids))==2;expected=query^order;rows.append(dict(pair=pair,split=split,format=fmt,query=query,family=fam,order=order,expected=expected,text=text,ids=ids,groups=[[x] for x in answerids],source_positions=positions,source_words=source,id=f'{pair}-{fmt}-{query}-{fam}'))
  if errors or len(rows)!=24:excluded.append({'pair':pair,'words':[a,b],'reasons':sorted(set(errors))});continue
  for fmt in ['colon','equals']:
   for order in [0,1]:assert len({tuple(r['source_positions']) for r in rows if r['format']==fmt and r['order']==order})==1
   for query in [0,1]:
    for fam in ['copy','binding','capital']:
     rr=[r for r in rows if r['format']==fmt and r['query']==query and r['family']==fam];assert len(rr[0]['ids'])==len(rr[1]['ids']);changed=[i for i,(u,v) in enumerate(zip(rr[0]['ids'],rr[1]['ids'])) if u!=v];assert changed==rr[0]['source_positions']
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
  for fmt in ['colon','equals']:
   for query in [0,1]:
    for split in ['discovery','transfer']:
     ix=[i for i,e in enumerate(examples) if (e['family'],e['format'],e['query'],e['split'])==(fam,fmt,query,split)];by=defaultdict(list)
     for i in ix:by[examples[i]['pair']].append(i)
     both=[all((measurements[i]['margin']>0)==(examples[i]['expected']==0) for i in inds) for inds in by.values()];top=[measurements[i]['top1'][examples[i]['expected']] for i in ix];c=dict(family=fam,format=fmt,query=query,split=split,pairs=len(by),both_correct=float(np.mean(both)) if both else 0.,full_vocab=float(np.mean(top)) if top else 0.);c['pass']=c['pairs']>=4 and c['both_correct']>=.75 and c['full_vocab']>=.5;cells.append(c)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_forward_max_abs':audit,'signed_AV_relative_error':reconstruction,'source_bus_same_token_max_abs':staticerr,'eligible_lexical_pairs':len(examples)//24,'all_source_positions_aligned':True,'only_source_tokens_change_with_order':True,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(torch.isfinite(v).all().item() for v in native.values())};assert audits['hashes_unchanged'] and audit<1e-4 and reconstruction<1e-4 and staticerr<1e-4 and audits['finite'];torch.save({'examples':examples,'native':native,'captures':captures,'bus':buscaps,'static_bus':static},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'competence':cells,'excluded':excluded,'measurements':measurements,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'competence':cells,'audits':audits}),flush=True)
if __name__=='__main__':main()

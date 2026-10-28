"""H893 one fixed agreement unit without family routing, mixed contexts."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from transformers import AutoTokenizer
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time,hashlib
S='bilin18_blind_number_unit';MODES=['exact','mean','rank1','rank7','random7','drop_cross'];NOUNS=[('pilot','pilots'),('nurse','nurses'),('singer','singers'),('soldier','soldiers'),('lawyer','lawyers'),('chef','chefs'),('scientist','scientists'),('engineer','engineers')]
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','compiled':ART/'bilin18_number_interaction_compression_compiled_units.pt','outputfits':ART/'bilin18_number_interaction_compression_fits.pt','natural':ART/'bilin18_frozen_featurizer_confirmation_captures.pt','manifest':ART/'bilin18_frozen_featurizer_confirmation_manifest.json'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H893','hashes':hashes,'modes':MODES,'axis':'A','fit_family':'task','natural_documents':128,'noun_pairs':NOUNS,'design':'One fixed H891 agreement axisA unit always applied, no family/template/gate selection. First128 natural confirmation docs at locked prefix positions plus8 predeclared nounpairs in newgrammar/copy templates. Fitcoefficients frozen. No outcome filtering. Lastquery L12 nativecross replacement, othercomponents exact.','freshness':'Synthetic noun/template combinations new to this unit investigation; natural documents reused in broader research, first test ofthisunit onthem, not globally untouched/pretraining-disjoint. Original reservedtest untouched.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples=[];excluded=[]
 for pair,(a,b) in enumerate(NOUNS):
  rows=[];errors=[]
  for fmt in ['G','H']:
   for family in ['grammar','copy']:
    for order,noun in enumerate([a,b]):
     text=(f'The {noun} behind the fence' if fmt=='G' else f'The {noun} beside the window') if family=='grammar' else (f'word {noun}; word {noun}; word' if fmt=='G' else f'item {noun}, item {noun}, item');answers=['is','are'] if family=='grammar' else [a,b];aid=[continuation_id(tok,text,' '+v) for v in answers]
     if any(v is None for v in aid):errors.append('answer_not_single');continue
     rows.append(dict(group=family,pair=pair,format=fmt,order=order,text=text,ids=tok.encode(text,add_special_tokens=False),answer_ids=aid,next_token=None))
  if errors:excluded.append(dict(pair=pair,reasons=errors));continue
  examples.extend(rows)
 panel=torch.load(paths['natural'],weights_only=False);manifest=json.loads(paths['manifest'].read_text());manifestok=True
 for i in range(128):
  row=panel['all_token_ids'][i];pos=panel['positions'][i];manifestok &= hashlib.sha256(row.numpy().astype(np.int64).tobytes()).hexdigest()==manifest['records'][i]['window_sha256_int64'];examples.append(dict(group='natural',document_index=i,document_hash=manifest['records'][i]['document_hash'],position=pos,ids=row[:pos+1].tolist(),next_token=int(row[pos+1]) if pos<127 else None))
 assert manifestok;(ART/f'{S}_manifest.json').write_text(json.dumps(dict(examples=examples,excluded=excluded),indent=2)+'\n');unit=torch.load(paths['compiled'],weights_only=False)['A','task'];fit=torch.load(paths['outputfits'],weights_only=False)['A','task'];model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 mod=model.blocks[12].mlp;L,R,D=[m.weight.double() for m in [mod.Left,mod.Right,mod.Down]];b=unit['number_axis'].cuda();origin=unit['number_origin'];mu,basis,random=[x.cuda() for x in fit];encoders={}
 for name,B in [('rank1',basis[:1]),('rank7',basis),('random7',random)]:
  aa=((B@D)*(R@b))@L+((B@D)*(L@b))@R;encoders[name]=(aa-(aa@b)[:,None]*b,origin*(aa@b)-B@mu,B)
 state={'mode':'baseline'};stash={};encodererr=0.
 def hook(mod,args,out):
  nonlocal encodererr
  if state['mode']=='baseline':return
  u=args[0][:,-1].double();t=u@b-origin;w=u-t[:,None]*b;C=((w@L.T)*(b@R.T)+(w@R.T)*(b@L.T))@D.T;mode=state['mode']
  if mode=='exact':hat=C
  elif mode=='mean':hat=mu.expand_as(C)
  elif mode=='drop_cross':hat=torch.zeros_like(C)
  else:
   E,off,B=encoders[mode];hat=mu+(u@E.T+off)@B;ref=mu+(C-mu)@B.T@B;encodererr=max(encodererr,float((hat-ref).norm()/ref.norm()))
  stash['t']=t.cpu();stash['Cerror']=((hat-C).norm(dim=-1)/C.norm(dim=-1)).cpu();z=out.clone();z[:,-1]+=(t[:,None]*(hat-C)).to(out);return z
 handle=mod.register_forward_hook(hook);buckets=defaultdict(list)
 for i,e in enumerate(examples):buckets[len(e['ids'])].append(i)
 records=[];actual=0;identity=0.
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),8):
    ix=ii[st:st+8];tokens=torch.tensor([examples[i]['ids'] for i in ix],device='cuda');state['mode']='baseline';base=final_logits(model,tokens).double().cpu();actual+=len(ix)
    for mode in MODES:
     state['mode']=mode;y=final_logits(model,tokens).double().cpu();actual+=len(ix)
     for n,i in enumerate(ix):
      e=examples[i];lp=torch.log_softmax(base[n],-1);lq=torch.log_softmax(y[n],-1);r=dict(index=i,group=e['group'],mode=mode,KL=float((lp.exp()*(lp-lq)).sum()),top1_changed=int(base[n].argmax())!=int(y[n].argmax()),coordinate=float(stash['t'][n]),interaction_relative_error=float(stash['Cerror'][n]))
      if e['next_token'] is not None:r.update(native_CE=float(-lp[e['next_token']]),edited_CE=float(-lq[e['next_token']]))
      if e['group']!='natural':
       ids=e['answer_ids'];sign=1 if e['order']==0 else -1;r.update(pair=e['pair'],format=e['format'],order=e['order'],native_margin=sign*float(base[n,ids[0]]-base[n,ids[1]]),edited_margin=sign*float(y[n,ids[0]]-y[n,ids[1]]),native_top1_correct=int(base[n].argmax())==ids[e['order']],edited_top1_correct=int(y[n].argmax())==ids[e['order']])
      if mode=='exact':identity=max(identity,float((y[n]-base[n]).abs().max()))
      records.append(r)
   print('DONE',length,actual,flush=True)
 finally:handle.remove()
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),natural_manifest_windows_verified=manifestok,exact_identity_logits_max_abs=identity,compiled_encoder_relative=encodererr,examples=len(examples),natural_documents=128,whole_model_forwards=actual,test_evaluations=0,finite=all(np.isfinite(r['KL']) for r in records));assert audits['hashes_unchanged'] and identity<1e-4 and encodererr<1e-10 and actual==len(examples)*7 and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,excluded=excluded,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

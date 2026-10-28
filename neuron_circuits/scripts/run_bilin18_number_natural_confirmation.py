"""H896 frozen ten-unit natural confirmation, all results retained."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time,hashlib
S='bilin18_number_natural_confirmation';MODES=['exact','replace','drop_cross']
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','units':ART/'bilin18_number_unit_recovery_units.pt','natural':ART/'bilin18_frozen_featurizer_confirmation_captures.pt','manifest':ART/'bilin18_frozen_featurizer_confirmation_manifest.json'};hashes={k:sha(p) for k,p in paths.items()};cfg=dict(experiment='H896',hashes=hashes,modes=MODES,document_indices=[128,512],design='All ten H894 units frozen. Each independently replaces final-query L12 cross term; exact and cross-removal controls. No unit selection, routing, gate fitting or refitting. Remaining384 locked natural documents, previously used in broader research but not these units. One locked position per document. Original reservedtest untouched.');(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');panel=torch.load(paths['natural'],weights_only=False);manifest=json.loads(paths['manifest'].read_text());examples=[]
 for i in range(128,512):
  row=panel['all_token_ids'][i];pos=int(panel['positions'][i]);assert hashlib.sha256(row.numpy().astype(np.int64).tobytes()).hexdigest()==manifest['records'][i]['window_sha256_int64'];examples.append(dict(document_index=i,document_hash=manifest['records'][i]['document_hash'],position=pos,ids=row[:pos+1].tolist(),next_token=int(row[pos+1]) if pos<len(row)-1 else None))
 assert len({e['document_hash'] for e in examples})==384
 (ART/f'{S}_manifest.json').write_text(json.dumps(dict(examples=examples),indent=2)+'\n');units={k:{n:v.cuda() for n,v in u.items()} for k,u in torch.load(paths['units'],weights_only=False).items()};assert len(units)==10;model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 mod=model.blocks[12].mlp;L,R,D=[m.weight.double() for m in [mod.Left,mod.Right,mod.Down]];state={'mode':'baseline'};stash={};encodererr=0.
 def hook(mod,args,out):
  nonlocal encodererr
  if state['mode']=='baseline':return
  U=units[state['unit']];u=args[0][:,-1].double();b=U['b'];t=u@b-U['origin'];w=u-t[:,None]*b;C=((w@L.T)*(b@R.T)+(w@R.T)*(b@L.T))@D.T;hat=U['mu']+(u@U['encoder']+U['offset'])[:,None]*U['v'];ref=U['mu']+((C-U['mu'])@U['v'])[:,None]*U['v'];encodererr=max(encodererr,float((hat-ref).norm()/ref.norm()));mode=state['mode']
  if mode=='exact':hat=C
  elif mode=='drop_cross':hat=torch.zeros_like(C)
  stash['coordinate']=t.cpu();stash['cross_norm']=(t[:,None]*C).norm(dim=-1).cpu();stash['error_norm']=(t[:,None]*(hat-C)).norm(dim=-1).cpu();z=out.clone();z[:,-1]+=(t[:,None]*(hat-C)).to(out);return z
 handle=mod.register_forward_hook(hook);buckets=defaultdict(list)
 for i,e in enumerate(examples):buckets[len(e['ids'])].append(i)
 records=[];actual=0;identity=0.
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),8):
    ix=ii[st:st+8];tokens=torch.tensor([examples[i]['ids'] for i in ix],device='cuda');state['mode']='baseline';base=final_logits(model,tokens).double().cpu();actual+=len(ix)
    for key in units:
     state['unit']=key
     for mode in MODES:
      state['mode']=mode;y=final_logits(model,tokens).double().cpu();actual+=len(ix)
      for n,i in enumerate(ix):
       e=examples[i];lp=torch.log_softmax(base[n],-1);lq=torch.log_softmax(y[n],-1);r=dict(index=i,document_index=e['document_index'],unit=key,mode=mode,KL=float((lp.exp()*(lp-lq)).sum()),top1_changed=int(base[n].argmax())!=int(y[n].argmax()),coordinate=float(stash['coordinate'][n]),cross_norm=float(stash['cross_norm'][n]),error_norm=float(stash['error_norm'][n]))
       if e['next_token'] is not None:r.update(native_CE=float(-lp[e['next_token']]),edited_CE=float(-lq[e['next_token']]))
       if mode=='exact':identity=max(identity,float((y[n]-base[n]).abs().max()))
       records.append(r)
   print('DONE',length,actual,flush=True)
 finally:handle.remove()
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),natural_manifest_windows_verified=True,exact_identity_logits_max_abs=identity,compiled_encoder_relative=encodererr,natural_documents=384,whole_model_forwards=actual,expected_forwards=384*31,test_evaluations=0,finite=all(np.isfinite(r['KL']) for r in records));assert audits['hashes_unchanged'] and identity<1e-4 and encodererr<1e-10 and actual==384*31 and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

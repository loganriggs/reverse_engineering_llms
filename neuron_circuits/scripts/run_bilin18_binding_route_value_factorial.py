"""H918: frozen-reader native endpoint factorization; no fitted features."""
from run_bilin18_multidomain_matched_competence import ROOT,ART,sha,SNAPSHOT,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_binding_route_value_factorial'
HEADS=[(8,3),(13,0),(14,4)]
MODES=['identity','routing','mixed_values','bus','current_values','full']
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'runner':Path(__file__),'panel':ART/'bilin18_multidomain_matched_competence_captures.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()}
 cfg={'experiment':'H918','hashes':hashes,'heads':HEADS,'modes':MODES,'design':'All 8 existing binding name pairs, 2 templates, task/copy, both endpoints. First4 discovery, last4 transfer; no fitting or selection. Native opposite endpoint donor, same family/template. Swap routing over all positions, mixed values, persistent bus only, current values only, or full head output. All other heads and downstream computation live. Local edge interventions, not global stream replacement.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];ix=[i for i,e in enumerate(ex) if e['domain']=='binding'];assert len(ix)==64
 lookup={(ex[i]['id'],ex[i]['order']):i for i in ix};pairs=sorted({ex[i]['pair'] for i in ix});buckets=defaultdict(list)
 for i in ix:buckets[len(ex[i]['ids'])].append(i)
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 deltas={};algerr=0.;averr=0.
 for i in ix:
  e=ex[i];j=lookup[e['id'],1-e['order']];assert len(e['ids'])==len(ex[j]['ids'])
  for l,h in HEADS:
   r=data['captures'][i][l,h];d=data['captures'][j][l,h];A=r['routing'].double();B=d['routing'].double();V=r['mixed_values'].double();W=d['mixed_values'].double();db=(data['bus'][j][:,h]-data['bus'][i][:,h]).double()*float(model.blocks[l].attn.lamb)
   base=(A[:,None]*V).sum(0);full=(B[:,None]*W).sum(0)-base;route=((B-A)[:,None]*V).sum(0);value=(A[:,None]*(W-V)).sum(0);bus=(A[:,None]*db).sum(0);current=(A[:,None]*(W-V-db)).sum(0);cross=((B-A)[:,None]*(W-V)).sum(0)
   algerr=max(algerr,float((full-route-value-cross).abs().max()),float((value-bus-current).abs().max()))
   averr=max(averr,float((base-r['head_output'].double()).norm()/r['head_output'].double().norm().clamp_min(1e-12)))
   for mode,v in [('identity',torch.zeros_like(base)),('routing',route),('mixed_values',value),('bus',bus),('current_values',current),('full',d['head_output'].double()-r['head_output'].double())]:deltas[i,l,h,mode]=v
 records=[];actual=0;replay=0.;identity=0.;saved={}
 for length,inds in sorted(buckets.items()):
  for st in range(0,len(inds),8):
   batch=inds[st:st+8];tokens=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');native=final_logits(model,tokens).double().cpu();actual+=len(batch)
   for n,i in enumerate(batch):replay=max(replay,float((native[n]-data['native'][i]).abs().max()))
   for l,h in HEADS:
    for mode in MODES:
     delta=torch.stack([deltas[i,l,h,mode] for i in batch]).cuda()
     def hook(mod,args):
      x=args[0].clone();x[:,-1,h*128:(h+1)*128]+=delta.to(x);return (x,)
     handle=model.blocks[l].attn.c_proj.register_forward_pre_hook(hook)
     try:y=final_logits(model,tokens).double().cpu()
     finally:handle.remove()
     actual+=len(batch)
     if mode=='identity':identity=max(identity,float((native-y).abs().max()))
     for n,i in enumerate(batch):
      e=ex[i];ids=[g[0] for g in e['groups']];o=e['expected'];margin=lambda z:float(z[ids[o]]-z[ids[1-o]])
      lp=native[n].log_softmax(-1);lq=y[n].log_softmax(-1)
      records.append({k:e[k] for k in ['pair','family','format','order']}|{'split':'discovery' if e['pair'] in pairs[:4] else 'transfer','layer':l,'head':h,'mode':mode,'native_margin':margin(native[n]),'edited_margin':margin(y[n]),'margin_drop':margin(native[n])-margin(y[n]),'KL':float((lp.exp()*(lp-lq)).sum()),'top1_correct':int(y[n].argmax())==ids[o]})
      saved[i,l,h,mode]=y[n]
   print('DONE',length,st,actual,flush=True)
 groups=defaultdict(list)
 for r in records:groups[r['split'],r['family'],r['layer'],r['head'],r['mode']].append(r)
 summary=[]
 for key,rr in groups.items():summary.append(dict(zip(['split','family','layer','head','mode'],key))|{'n':len(rr),'mean_margin_drop':float(np.mean([r['margin_drop'] for r in rr])),'rms_margin_change':float(np.mean([r['margin_drop']**2 for r in rr])**.5),'mean_KL':float(np.mean([r['KL'] for r in rr])),'top1':float(np.mean([r['top1_correct'] for r in rr]))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_replay_max_abs':replay,'identity_max_abs':identity,'factorization_max_abs':algerr,'cached_AV_relative_error':averr,'whole_model_forwards':actual,'expected_forwards':64*19,'test_evaluations':0,'finite':all(np.isfinite(r['KL']) and np.isfinite(r['margin_drop']) for r in records)}
 assert audits['hashes_unchanged'] and replay<1e-4 and identity==0 and algerr<1e-10 and averr<1e-4 and actual==64*19 and audits['finite'],audits
 torch.save(saved,ART/f'{S}_logits.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

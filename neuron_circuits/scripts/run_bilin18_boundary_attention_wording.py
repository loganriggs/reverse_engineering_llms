"""H659: discovery-only finite boundary and attention prediction."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
STEM='bilin18_boundary_attention_wording'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_operation_paraphrase_panel_manifest.json','captures':ART/'bilin18_operation_paraphrase_panel_captures.pt','original_manifest':ART/'bilin18_operation_control_panel_manifest.json','original_captures':ART/'bilin18_operation_control_panel_captures.pt','exact':ART/'bilin18_finite_factorial_captures.pt','previous':ART/'bilin18_finite_factorial.json','runner':Path(__file__),'prior_prediction':ART/'bilin18_boundary_attention_prediction_captures.pt'};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());oldrows=json.loads(paths['original_manifest'].read_text());cap=torch.load(paths['captures'],weights_only=False);oldcap=torch.load(paths['original_captures'],weights_only=False);exact=torch.load(paths['exact'],weights_only=False);source,target=exact['source'],exact['target'];lookup={(r['family'],r['control'],r['item'],r['form']):i for i,r in enumerate(rows)};families=sorted({r['family'] for r in rows})
 for a,b in zip(oldrows,rows):
  for k in ['id','input','expected','split']:assert a[k]==b[k]
 cfg=dict(experiment='H660',hashes=hashes,design='Original/new-wording discovery means crossed3x3 with exact component controls. Four discovery items only; same96 heldout items. Copy-to-task block9 state and attention10-17 output changes, four items per family/form. Evaluate96 second-wording transfer copy prompts. Native/predicted/exact3x3 component factorial; other-task pair control and live-additive attention variant. Predicted attention replaces writes with cached receiver-native attention plus discovery mean, while native mode recomputes live. No MLP edits; full strength fixed.',limitations='Predictions require receiver native baseline, known family/form and full network. Exact controls use donor states. Retrospective panel, no output fitting. Fixed write prediction differs from live additive feedback.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 attnevals=0
 def attention(c,r,indices):
  nonlocal attnevals
  result={l:{} for l in range(10,18)};buckets=defaultdict(list)
  for i in indices:buckets[len(r[i]['ids'])].append(i)
  for l in range(10,18):
   block=model.blocks[l]
   for ii in buckets.values():
    for st in range(0,len(ii),16):
     bb=ii[st:st+16];x=torch.stack([c['caches'][l][i] for i in bb]).cuda();x0=torch.stack([c['x0'][i] for i in bb]).cuda();bus=torch.stack([c['bus'][i] for i in bb]).cuda();a,_=block.attn(F.rms_norm(block.lambdas[0]*x+block.lambdas[1]*x0,(1152,)),bus);attnevals+=len(bb)
     for k,i in enumerate(bb):result[l][i]=a[k,-1].cpu()
  return result
 discovery=[i for i,r in enumerate(oldrows) if r['item']<4 and r['control'] in ['copy','task']];da=attention(oldcap,oldrows,discovery);ra=attention(cap,rows,source);means={}
 for fam in families:
  for form in range(4):
   ss=[lookup[fam,'copy',i,form] for i in range(4)];tt=[lookup[fam,'task',i,form] for i in range(4)];b=(oldcap['native_captures'][9][tt].double()-oldcap['native_captures'][9][ss].double()).mean(0);a=torch.stack([torch.stack([da[l][t].double()-da[l][s].double() for l in range(10,18)]) for s,t in zip(ss,tt)]).mean(0);means[fam,form]=dict(boundary=b,attention=a)
 newda=attention(cap,rows,discovery);newmeans={}
 for fam in families:
  for form in range(4):
   ss=[lookup[fam,'copy',i,form] for i in range(4)];tt=[lookup[fam,'task',i,form] for i in range(4)];b=(cap['native_captures'][9][tt].double()-cap['native_captures'][9][ss].double()).mean(0);aa=torch.stack([torch.stack([newda[l][t].double()-newda[l][s].double() for l in range(10,18)]) for s,t in zip(ss,tt)]).mean(0);newmeans[fam,form]=dict(boundary=b,attention=aa)
 locked=ART/f'{STEM}_locked.pt';torch.save(dict(original=means,new=newmeans),locked);paths['locked']=locked;hashes['locked']=hashlib.sha256(locked.read_bytes()).hexdigest();cfg['hashes']=hashes;(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));predb=[];preda=[];otherb=[];othera=[]
 for s in source:
  r=rows[s];m=means[r['family'],r['form']];predb.append(m['boundary']);preda.append(m['attention']);other=[means[f,r['form']] for f in families if f!=r['family']];otherb.append(torch.stack([x['boundary'] for x in other]).mean(0));othera.append(torch.stack([x['attention'] for x in other]).mean(0))
 predb,preda,otherb,othera=map(torch.stack,[predb,preda,otherb,othera]);baseb=cap['native_captures'][9][source];basea=torch.stack([torch.stack([ra[l][s] for l in range(10,18)]) for s in source]);donorb=cap['native_captures'][9][target];donora=torch.stack([exact['donor_attention'][l] for l in range(10,18)],1);buckets=defaultdict(list)
 for i,s in enumerate(source):buckets[len(rows[s]['ids'])].append(i)
 newb=torch.stack([newmeans[rows[s]['family'],rows[s]['form']]['boundary'] for s in source]);newa=torch.stack([newmeans[rows[s]['family'],rows[s]['form']]['attention'] for s in source]);forwards=0
 def evaluate(bmode,amode):
  nonlocal forwards
  logits=torch.empty(96,model.lm_head.weight.shape[0])
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];hooks=[]
    if bmode!='native':
     def boundary(mod,args,out):
      y,bus=out;y=y.clone();v=donorb[bb] if bmode=='exact' else baseb[bb]+(newb if bmode=='new' else predb)[bb];y[:,-1]=v.to(y);return y,bus
     hooks.append(model.blocks[9].register_forward_hook(boundary))
    if amode!='native':
     for l in range(10,18):
      def attedit(mod,args,out,l=l):
       y,bus=out;y=y.clone()
       if amode=='exact':v=donora[bb,l-10]
       elif amode=='live':v=y[:,-1]+preda[bb,l-10].to(y)
       else:v=basea[bb,l-10]+(newa if amode=='new' else preda)[bb,l-10]
       y[:,-1]=v.to(y);return y,bus
      hooks.append(model.blocks[l].attn.register_forward_hook(attedit))
    try:z,_=run(model,torch.tensor([rows[source[i]]['ids'] for i in bb],device='cuda'));logits[bb]=z.cpu()
    finally:
     for h in hooks:h.remove()
    forwards+=len(bb)
  return logits
 tasklog=exact['donor_logits'].double().log_softmax(-1);p=tasklog.exp();records=[];outputs={};conditions=[(b,a) for b in ['predicted','new','exact'] for a in ['predicted','new','exact']]+[('native','native'),('native','exact'),('exact','native')]
 for bmode,amode in conditions:
  z=evaluate(bmode,amode);outputs[bmode,amode]=z;lp=z.double().log_softmax(-1);kl=(p*(tasklog-lp)).sum(-1)
  for i,s in enumerate(source):
   r=rows[s];t=rows[target[i]];ids=t['candidate_ids'];aid=t['answer_id'];records.append(dict(id=r['id'],family=r['family'],form=r['form'],item=r['item'],boundary=bmode,attention=amode,task_kl=float(kl[i]),target_ce=float(-lp[i,aid]),candidate_correct=ids[int(z[i,ids].argmax())]==aid,top1_correct=int(z[i].argmax())==aid))
  print('COMPLETE',bmode,amode,flush=True)
 replays={str((b,a)):float((outputs[b,a]-exact['logits'][m]).abs().max()) for b,a,m in [('native','native',0),('exact','native',1),('native','exact',2),('exact','exact',3)]};assert max(replays.values())<1e-3;prior=torch.load(paths['prior_prediction'],weights_only=False)['logits'];prior_diff=max(float((outputs[b,a]-prior[b,a]).abs().max()) for b,a in [('predicted','predicted'),('predicted','exact'),('exact','predicted')]);assert prior_diff==0;audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},whole_model_forwards=forwards,previous_prediction_max_abs=prior_diff,cached_attention_evaluations=attnevals,exact_factorial_replay_max_abs=replays,finite=all(np.isfinite(r['task_kl']) and np.isfinite(r['target_ce']) for r in records));torch.save(dict(logits=outputs,means=means,source=source,target=target),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()

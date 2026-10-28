"""H976 locked MLP3 composition/period4 rescue on prospective panel."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from pathlib import Path
from collections import defaultdict
import torch,json,numpy as np,time
S='bilin18_prospective_route_transfer'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'data':ART/'bilin18_prospective_composition_panel_captures.pt','panel':ART/'bilin18_prospective_composition_panel.json','model':SNAPSHOT/'pytorch_model.bin','forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py'};hashes={k:sha(p) for k,p in paths.items()};d=torch.load(paths['data'],weights_only=False);panel=json.loads(paths['panel'].read_text());ex=d['examples'];qs=d['quartets'];cap=d['captures'];native=d['native']
 cfg={'experiment':'H976','hashes':hashes,'design':'Frozen adjective-last-subtoken MLP3 local/total factorial intervention formulas and periodMLP4 native write rescue from H973/974. Fit no sites or encoder. Recompute quartet algebra from new teacher activations; this tests site/formula transfer, not a frozen learned featurizer. All64 quartets included; passing native cohorts interpreted separately.','controls':'Identity, +/- norm-matched random upstream factorial edits for local and total; +/- norm-matched random period correction for each ablation; native-period rescue identity.','metrics':'Valence-aligned margin loss and native-reference KL; recovery of ablation loss after period rescue, KL remaining. Multi-subtoken adjectives patched only at last subtoken, per frozen convention.','limits':'Small cohort sizes, teacher-informed and potentially off-manifold edits, shared lexicons across templates. No training or reserved-test access.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 m=model.blocks[3].mlp;L=m.Left.weight.detach().double();R=m.Right.weight.detach().double();D=m.Down.weight.detach().double();bias=m.Down_bias.detach().double();H=torch.tensor([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],dtype=torch.float64,device='cuda');parts={};writeerr=0.;algebra=0.
 def B(u,v):return ((u@L.T)*(v@R.T))@D.T
 for q in qs:
  X=torch.stack([cap[i]['normalized'] for i in q['indices']]).double().cuda();C=H.T@X/4;Y=B(X,X)+bias;local=B(C[1],C[2])+B(C[2],C[1]);inherited=B(C[0],C[3])+B(C[3],C[0]);total=local+inherited;write=torch.stack([cap[i]['adjective_write'] for i in q['indices']]).double().cuda();writeerr=max(writeerr,float((Y-write).norm()/write.norm()));algebra=max(algebra,float((total-(H.T@Y/4)[3]).norm()/total.norm()));parts[q['id']]={'local':local.float().cpu(),'total':total.float().cpu()}
 jobs=[];normerr=0.
 for qi,q in enumerate(qs):
  for state,i in enumerate(q['indices']):
   arms={'none':torch.zeros(1152)}
   for ci,c in enumerate(['local','total']):
    vec=parts[q['id']][c];g=torch.Generator().manual_seed(976000+qi*2+ci);z=torch.randn(1152,generator=g);z=z/z.norm()*vec.norm();normerr=max(normerr,float(abs(z.norm()-vec.norm())/vec.norm().clamp_min(1e-20)));v=ex[i]['valence'];arms[c]=-v*vec;arms['random_'+c+'_plus']=v*z;arms['random_'+c+'_minus']=-v*z
   for ablation,edit in arms.items():
    for rescue in (['none','restore','random_plus','random_minus'] if ablation in ['none','local','total'] else ['none']):jobs.append(dict(q=q,state=state,i=i,ablation=ablation,edit=edit,rescue=rescue))
 current=[]
 def ablate(module,args,out):
  y=out.clone()
  for k,j in enumerate(current):
   if j['ablation']!='none':i=j['i'];y[k,ex[i]['adjective_position']]=cap[i]['adjective_write'].to(y)+j['edit'].to(y)
  return y
 def rescue(module,args,out):
  nonlocal normerr
  y=out.clone()
  for k,j in enumerate(current):
   if j['rescue']=='none':continue
   i=j['i'];t=ex[i]['period_position'];delta=cap[i]['period_write'].to(y)-out[k,t]
   if j['rescue']=='restore':y[k,t]+=delta
   else:
    g=torch.Generator().manual_seed(976500+i);z=torch.randn(1152,generator=g).to(y);z=z/z.norm()*delta.norm();normerr=max(normerr,float(abs(z.norm()-delta.norm())/delta.norm().clamp_min(1e-20)));y[k,t]+=z if j['rescue']=='random_plus' else -z
  return y
 hooks=[model.blocks[3].mlp.register_forward_hook(ablate),model.blocks[4].mlp.register_forward_hook(rescue)];actual=0;identity=0.;records=[];buckets=defaultdict(list)
 for j in jobs:buckets[len(ex[j['i']]['ids'])].append(j)
 try:
  for length,jj in sorted(buckets.items()):
   for st in range(0,len(jj),24):
    current=jj[st:st+24];out,_=forward(model,torch.tensor([ex[j['i']]['ids'] for j in current],device='cuda'));actual+=len(current)
    for k,j in enumerate(current):
     y=out[k].double().cpu();base=native[j['i']];lp=y.log_softmax(0);ln=base.log_softmax(0);margin=measure(y,ex[j['i']]['groups'])['margin'];bm=measure(base,ex[j['i']]['groups'])['margin'];v=ex[j['i']]['valence']
     if j['ablation']=='none':identity=max(identity,float((y-base).abs().max()))
     records.append({key:j['q'][key] for key in ['id','format','cohort','source']}|dict(state=j['state'],ablation=j['ablation'],rescue=j['rescue'],margin=margin,loss=v*(bm-margin),KL=float((ln.exp()*(ln-lp)).sum()),binary_correct=v*margin>0))
   print('DONE',length,actual,flush=True)
 finally:
  for h in hooks:h.remove()
 grouped=defaultdict(list)
 for r in records:grouped[r['format'],r['cohort'],r['ablation'],r['rescue']].append(r)
 summary=[]
 for key,rr in grouped.items():
  loss=float(np.mean([r['loss'] for r in rr]));kl=float(np.mean([r['KL'] for r in rr]));ref=grouped[key[0],key[1],key[2],'none'];bl=float(np.mean([r['loss'] for r in ref]));bk=float(np.mean([r['KL'] for r in ref]));summary.append(dict(zip(['format','cohort','ablation','rescue'],key))|dict(n=len(rr),mean_loss=loss,rms_loss=float(np.mean([r['loss']**2 for r in rr])**.5),mean_KL=kl,margin_recovery=(bl-loss)/bl if key[2] in ['local','total'] else None,KL_remaining=kl/bk if key[2] in ['local','total'] else None))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),whole_model_forwards=actual,expected_forwards=len(jobs),identity_logits_max_abs=identity,write_replay_relative=writeerr,interaction_identity_relative=algebra,random_norm_relative=normerr,finite=all(np.isfinite(r['KL']) for r in records),no_training=True,test_evaluations=0)
 assert audits['hashes_unchanged'] and actual==len(jobs) and identity<1e-4 and writeerr<1e-5 and algebra<1e-10 and normerr<1e-5 and audits['finite']
 torch.save(dict(parts=parts),ART/f'{S}_parts.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,competence=panel['summary'],records=records,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

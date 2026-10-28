"""H971 finite removal of attention5 interaction components."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import measure
from run_bilin18_task_bank_attention_sites import forward,manual_patch
from pathlib import Path
from collections import defaultdict
import torch,json,numpy as np,time
S='bilin18_attention5_component_removals'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'sources':ART/'bilin18_attention5_sources_captures.pt','data':ART/'bilin18_factorial_valence_negation_captures.pt','trace':ART/'bilin18_valence_interaction_trace_captures.pt','panel':ART/'bilin18_factorial_valence_negation.json','model':SNAPSHOT/'pytorch_model.bin','forward':ROOT/'scripts/run_bilin18_task_bank_attention_sites.py'};hashes={k:sha(p) for k,p in paths.items()};d=torch.load(paths['sources'],weights_only=False);data=torch.load(paths['data'],weights_only=False);trace=torch.load(paths['trace'],weights_only=False);panel=json.loads(paths['panel'].read_text());ex=data['examples'];qs=data['quartets'];native=data['native'];W=d['projection'];components={};sumerr=0.
 names=['total','value','routing','cross','adjective','period','other_suffix','prefix','negation']+[f'head{h}' for h in range(9)]
 cfg={'experiment':'H971','hashes':hashes,'design':'At attention5 final-token write subtract a*n times each locked factorial component: total, transported value, routing, cross, source roles and all9 heads. Native bus unchanged; downstream computation recomputed. Each component has paired norm-matched random factorial edits, plus a common identity arm.','source_roles':'Period is suffix:0 (verified punctuation token in manifest); other suffix includes remaining trailing tokens. Role/head terms are summed native P*V interactions, hence independent of missing-slot value-padding convention. Value/routing/cross inherit H970 padding convention.','metrics':'Native-aligned answer-margin loss, KL(native||edited); KL(full-interaction-removal||component-removal) normalized by KL(full-removal||native) to assess whether component removal reproduces the full intervention.','scope':'Same24 locked quartets; only familiar formatA passes native gate. Exploratory same-panel decomposition, no new generalization claim. These are output component removals, not direct routing or value modifications inside attention.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 for q in qs:
  vv={name:torch.zeros(1152,dtype=torch.float64) for name in names};vv.update({name:d['parts'][q['id']][name] for name in ['total','value','routing','cross']})
  for state,i in enumerate(q['indices']):
   P=d['captures'][i]['P'].double();V=d['captures'][i]['V'].double();Z=torch.einsum('ht,thd,ohd->tho',P,V,W);sign=ex[i]['valence']/4
   for t,slot in enumerate(d['slots'][i]):
    role=slot.split(':')[0];role='period' if slot=='suffix:0' else ('other_suffix' if role=='suffix' else role);vv[role]+=sign*Z[t].sum(0)
    for head in range(9):vv[f'head{head}']+=sign*Z[t,head]
  for keys in [['value','routing','cross'],['adjective','period','other_suffix','prefix','negation'],[f'head{h}' for h in range(9)]]:sumerr=max(sumerr,float((sum(vv[k] for k in keys)-vv['total']).norm()/vv['total'].norm()))
  components[q['id']]={k:v.float() for k,v in vv.items()}
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 refs=[];jobs=[];normerr=0.
 for qi,q in enumerate(qs):
  for state,i in enumerate(q['indices']):
   b=dict(q=q,state=state,i=i);v=ex[i]['valence'];refs.append(b|dict(component='total',arm='remove',edit=-v*components[q['id']]['total']));jobs.append(b|dict(component='identity',arm='identity',edit=torch.zeros(1152)))
   for ci,name in enumerate(names):
    c=components[q['id']][name];gen=torch.Generator().manual_seed(971000+qi*100+ci);z=torch.randn(1152,generator=gen);z=z/z.norm()*c.norm();normerr=max(normerr,float(abs(z.norm()-c.norm())/c.norm().clamp_min(1e-20)))
    if name!='total':jobs.append(b|dict(component=name,arm='remove',edit=-v*c))
    jobs.extend([b|dict(component=name,arm='random_plus',edit=v*z),b|dict(component=name,arm='random_minus',edit=-v*z)])
 records=[];full={};actual=0;ident=0.
 def execute(jj,reference=False):
  nonlocal actual,ident
  buckets=defaultdict(list)
  for j in jj:buckets[len(ex[j['i']]['ids'])].append(j)
  for length,seq in sorted(buckets.items()):
   for st in range(0,len(seq),24):
    batch=seq[st:st+24];rep=torch.stack([trace['captures'][j['i']][5,'attention']+j['edit'] for j in batch]);out,_=forward(model,torch.tensor([ex[j['i']]['ids'] for j in batch],device='cuda'),patch=(5,rep));actual+=len(batch)
    for k,j in enumerate(batch):
     y=out[k].double().cpu();base=native[j['i']]
     if reference:full[j['i']]=y
     ref=full[j['i']];lp=y.log_softmax(0);ln=base.log_softmax(0);lr=ref.log_softmax(0);margin=measure(y,ex[j['i']]['groups'])['margin'];bm=measure(base,ex[j['i']]['groups'])['margin'];v=ex[j['i']]['valence']
     if j['arm']=='identity':ident=max(ident,float((y-base).abs().max()))
     records.append({key:j['q'][key] for key in ['id','format','source','cohort']}|dict(state=j['state'],arm=j['arm'],component=j['component'],margin=margin,loss=v*(bm-margin),KL=float((ln.exp()*(ln-lp)).sum()),full_KL=float((lr.exp()*(lr-lp)).sum()),full_baseline_KL=float((lr.exp()*(lr-ln)).sum()),binary_correct=v*margin>0))
 execute(refs,True);execute(jobs)
 i=qs[0]['indices'][0];rep=(trace['captures'][i][5,'attention']-components[qs[0]['id']]['value']).unsqueeze(0);ids=torch.tensor([ex[i]['ids']],device='cuda');a,_=forward(model,ids,patch=(5,rep));b=manual_patch(model,ids,5,rep);manualerr=float((a-b).abs().max());actual+=2
 grouped=defaultdict(list)
 for r in records:grouped[r['format'],r['cohort'],r['component'],r['arm']].append(r)
 summary=[]
 for key,rr in grouped.items():summary.append(dict(zip(['format','cohort','component','arm'],key))|dict(n=len(rr),mean_loss=float(np.mean([r['loss'] for r in rr])),rms_loss=float(np.mean([r['loss']**2 for r in rr])**.5),mean_KL=float(np.mean([r['KL'] for r in rr])),full_KL_ratio=sum(r['full_KL'] for r in rr)/sum(r['full_baseline_KL'] for r in rr),binary_correct=float(np.mean([r['binary_correct'] for r in rr]))))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),whole_model_forwards=actual,expected_forwards=len(refs)+len(jobs)+2,identity_logits_max_abs=ident,component_sum_relative=sumerr,random_norm_relative=normerr,independent_manual_patch_max_abs=manualerr,finite=all(np.isfinite(r['KL']) for r in records),no_training=True,test_evaluations=0)
 assert audits['hashes_unchanged'] and actual==audits['expected_forwards'] and ident<1e-4 and sumerr<1e-10 and normerr<1e-5 and manualerr<1e-4 and audits['finite']
 torch.save(dict(components=components),ART/f'{S}_components.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,competence=panel['competence'],records=records,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print([r for r in summary if r['format']=='A' and r['cohort']=='familiar_adjectives' and r['arm']=='remove'],audits,flush=True)
if __name__=='__main__':main()

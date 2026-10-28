"""H965: discovery-only rank-one task directions and cross-task causal transfer."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from pathlib import Path
from collections import defaultdict
import torch,json,numpy as np,time
S='bilin18_shared_task_directions'
FAMILIES=['agreement','sentiment','negation','pronoun_gender']
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'runner':Path(__file__),'screen':ART/'bilin18_expanded_task_bank.json','examples':ART/'bilin18_expanded_task_bank_captures.pt','writes':ART/'bilin18_task_bank_causal_sites_captures.pt','model':SNAPSHOT/'pytorch_model.bin','forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py'};hashes={k:sha(p) for k,p in paths.items()}
 screen=json.loads(paths['screen'].read_text());data=torch.load(paths['examples'],weights_only=False);ex=data['examples'];native=data['native'];writes=torch.load(paths['writes'],weights_only=False)['writes'];rows=[r for r in screen['records'] if r['family'] in FAMILIES and r['same_final_token']];assert len(rows)==64 and not any(r['whole_task_holdout'] for r in rows)
 cfg={'experiment':'H965','hashes':hashes,'families':FAMILIES,'layer':12,'design':'Normalize mean discovery MLP write difference endpoint1 minus endpoint0 for each family. Freeze four rank-one directions before validation. For every target pair project its true donor delta onto each source direction; test projected edit and complementary edit separately. This is teacher-informed span adequacy, not an encoder or independently predicted intervention.','controls':'Identity, complete donor swap, fixed random rank-one projection, per-source +/- Gaussian edits matched to projected norm. Privileged generic readout control projects onto mean unembedding weight group1 minus group0 (not the exact downstream Jacobian).','metrics':'Signed donor task-margin change; KL(full donor patch||candidate patch) and KL(native||candidate). Fidelity is relative to same-recipient full-write patch, not natural donor logits.','splits':'Only discovery rows fit directions. Candidate lexical cohorts are not strictly vocabulary disjoint. Alternate-format native gate failures remain diagnostics. No whole-task training holdouts or reserved tests accessed.','hypotheses':['All four share a direction','Only sentiment and negation share a direction','Each task has a separate rank-one direction','Low rank predicts margins but misses distribution changes','A direct output-label contrast explains apparent transfer','Mean directions miss meaningful context-dependent subspaces'],'limits':'Four discovery pairs per family; supervised directions; exact donor delta used for projection coefficients; no claim of unsupervised manifold or shared upstream computation.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 directions={};fitids={};coherence={}
 for family in FAMILIES:
  rr=[r for r in rows if r['family']==family and r['split']=='discovery'];assert len(rr)==4
  D=torch.stack([writes[r['indices'][1]][12]-writes[r['indices'][0]][12] for r in rr]).double();u=D.mean(0);coherence[family]=float(u.norm()/D.norm(dim=1).mean());directions[family]=(u/u.norm()).float();fitids[family]=[r['id'] for r in rr]
 gen=torch.Generator().manual_seed(965);u=torch.randn(1152,generator=gen);directions['random']=u/u.norm()
 frozen=ART/f'{S}_directions.pt';torch.save(dict(directions=directions,fitids=fitids,coherence=coherence),frozen);frozenhash=sha(frozen)
 cos={a:{b:float(directions[a]@directions[b]) for b in FAMILIES} for a in FAMILIES}
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 W=model.lm_head.weight.detach().cpu().float();records=[];full={};actual=0;identity=0.;projerr=0.;normerr=0.
 def execute(jobs,reference=False):
  nonlocal actual,identity
  buckets=defaultdict(list)
  for j in jobs:buckets[len(ex[j['i']]['ids'])].append(j)
  for length,jj in sorted(buckets.items()):
   for st in range(0,len(jj),24):
    batch=jj[st:st+24];ids=torch.tensor([ex[j['i']]['ids'] for j in batch],device='cuda');repl=torch.stack([writes[j['i']][12]+j['edit'] for j in batch]);out,_=forward(model,ids,patch=(12,repl));actual+=len(batch)
    for k,j in enumerate(batch):
     y=out[k].double().cpu();key=j['r']['id'],j['rec'];base=native[j['i']];lp=y.log_softmax(0);ln=base.log_softmax(0)
     if reference:full[key]=y
     ref=full[key];lr=ref.log_softmax(0);bm=measure(base,ex[j['i']]['groups'])['margin'];ym=measure(y,ex[j['i']]['groups'])['margin'];fm=measure(ref,ex[j['i']]['groups'])['margin'];sign=1 if j['rec']==1 else -1
     if j['arm']=='identity':identity=max(identity,float((y-base).abs().max()))
     records.append({a:j['r'][a] for a in ['id','family','split','source','format']}|dict(recipient=j['rec'],arm=j['arm'],direction=j['direction'],effect=sign*(ym-bm),full_effect=sign*(fm-bm),margin_error=ym-fm,KL_full_to_candidate=float((lr.exp()*(lr-lp)).sum()),KL_native_to_candidate=float((ln.exp()*(ln-lp)).sum()),KL_full_to_native=float((lr.exp()*(lr-ln)).sum()),edit_norm=float(j['edit'].norm()),donor_norm=j['donor_norm']))
 refs=[];jobs=[]
 for rid,r in enumerate(rows):
  for rec in [0,1]:
   i=r['indices'][rec];don=r['indices'][1-rec];delta=writes[don][12]-writes[i][12];base=dict(r=r,rec=rec,i=i,donor_norm=float(delta.norm()));refs.append(base|dict(arm='full',direction='full',edit=delta));jobs.append(base|dict(arm='identity',direction='identity',edit=torch.zeros_like(delta)))
   g=ex[i]['groups'];u=W[g[1]].mean(0)-W[g[0]].mean(0);u=u/u.norm();local=directions|{'readout':u}
   for name,u in local.items():
    p=(delta@u)*u;residual=delta-p;projerr=max(projerr,float(abs(p@residual)/(delta.norm().square().clamp_min(1e-20))))
    jobs.append(base|dict(arm='projection',direction=name,edit=p))
    if name in FAMILIES:
     jobs.append(base|dict(arm='complement',direction=name,edit=residual))
     gen=torch.Generator().manual_seed(965000+rid*10+rec);z=torch.randn(1152,generator=gen);z=z/z.norm()*p.norm();normerr=max(normerr,float(abs(z.norm()-p.norm())/p.norm().clamp_min(1e-20)))
     for sign in [-1,1]:jobs.append(base|dict(arm='random_plus' if sign==1 else 'random_minus',direction=name,edit=sign*z))
 execute(refs,True);execute(jobs)
 grouped=defaultdict(list)
 for r in records:grouped[r['family'],r['split'],r['arm'],r['direction']].append(r)
 summary=[]
 for key,rr in grouped.items():
  den=sum(r['KL_full_to_native'] for r in rr);summary.append(dict(zip(['family','split','arm','direction'],key))|dict(n=len(rr),effect=float(np.mean([r['effect'] for r in rr])),rms_effect=float(np.mean([r['effect']**2 for r in rr])**.5),full_effect=float(np.mean([r['full_effect'] for r in rr])),margin_error_rms=float(np.mean([r['margin_error']**2 for r in rr])**.5),KL_ratio=sum(r['KL_full_to_candidate'] for r in rr)/den,mean_KL=float(np.mean([r['KL_full_to_candidate'] for r in rr])),mean_edit_norm=float(np.mean([r['edit_norm'] for r in rr]))))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),frozen_directions_unchanged=sha(frozen)==frozenhash,identity_logits_max_abs=identity,projection_orthogonality_relative=projerr,random_norm_relative=normerr,whole_model_forwards=actual,expected_forwards=len(refs)+len(jobs),finite=all(np.isfinite(r['KL_full_to_candidate']) for r in records),test_evaluations=0,whole_task_holdouts_accessed=False)
 assert audits['hashes_unchanged'] and audits['frozen_directions_unchanged'] and identity<1e-4 and max(projerr,normerr)<1e-5 and actual==len(refs)+len(jobs) and audits['finite']
 (ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,summary=summary,records=records,direction_cosines=cos,coherence=coherence,fitids=fitids,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(cos,audits,flush=True)
if __name__=='__main__':main()

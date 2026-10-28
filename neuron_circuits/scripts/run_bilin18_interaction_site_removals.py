"""H969: finite factorial-interaction removals through every write site."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward as mlp_forward,measure
from run_bilin18_task_bank_attention_sites import forward as attention_forward
from pathlib import Path
from collections import defaultdict
import torch,json,numpy as np,time
S='bilin18_interaction_site_removals'
@torch.inference_mode()
def manual(model,ids,layer,kind,replacement):
 x0=torch.nn.functional.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
 for l,b in enumerate(model.blocks):
  raw=b.lambdas[0]*x+b.lambdas[1]*x0;att,bus=b.attn(torch.nn.functional.rms_norm(raw,(1152,)),bus)
  if l==layer and kind=='attention':att=att.clone();att[:,-1]=replacement.to(att)
  pre=raw+att;m=b.mlp(torch.nn.functional.rms_norm(pre,(1152,)))
  if l==layer and kind=='mlp':m=m.clone();m[:,-1]=replacement.to(m)
  x=pre+m
 return 30*torch.tanh(model.lm_head(torch.nn.functional.rms_norm(x,(1152,))[:,-1]).float()/30)
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'trace':ART/'bilin18_valence_interaction_trace_captures.pt','data':ART/'bilin18_factorial_valence_negation_captures.pt','panel':ART/'bilin18_factorial_valence_negation.json','model':SNAPSHOT/'pytorch_model.bin','mlp_forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','attention_forward':ROOT/'scripts/run_bilin18_task_bank_attention_sites.py'};hashes={k:sha(p) for k,p in paths.items()};trace=torch.load(paths['trace'],weights_only=False);data=torch.load(paths['data'],weights_only=False);panel=json.loads(paths['panel'].read_text());ex=data['examples'];qs=data['quartets'];native=data['native'];cap=trace['captures'];co=trace['coefficients']
 cfg={'experiment':'H969','hashes':hashes,'design':'Each of18 attention and18 MLP final-token output sites, each locked quartet: subtract a*n times its native factorial interaction coefficient. Recompute current/downstream computation naturally, preserving native attention bus. Identity and paired +/- isotropic random factorial edits matched to removed interaction norm. Random direction fixed per quartet/site and shared across its four states, preserving additive factor coefficients.','selection':'All sites evaluated; rank mean valence-aligned margin loss on familiar format A only. Other five cohorts remain native-failed diagnostics, not independent successful validation. No fitted featurizer or prediction from unseen labels.','limits':'Four-state teacher-informed removal and all-site screening are exploratory. Removing output interaction is not a natural semantic intervention. Positive loss can reflect disruption; random controls constrain but do not eliminate that interpretation. No per-example competence filtering.','metrics':'Native-minus-edited answer margin times intended binary valence; native-reference full-vocabulary KL; binary endpoint accuracy. Recompute quartet margin interaction coefficient to verify factorial accounting.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 actual=0;ident=0.;normerr=0.;factorerr=0.;records=[];manualerr=0.
 for kind in ['attention','mlp']:
  fn=attention_forward if kind=='attention' else mlp_forward
  for layer in range(18):
   buckets=defaultdict(list)
   for qi,q in enumerate(qs):
    interaction=co[q['id']][layer,kind][3].float();gen=torch.Generator().manual_seed(969000+qi*100+layer*2+(kind=='mlp'));noise=torch.randn(1152,generator=gen);noise=noise/noise.norm()*interaction.norm();normerr=max(normerr,float(abs(noise.norm()-interaction.norm())/interaction.norm().clamp_min(1e-20)))
    # Algebraic removal cancels the interaction coefficient without changing constant/main effects.
    ys=torch.stack([cap[i][layer,kind] for i in q['indices']]).double();v=torch.tensor([ex[i]['valence'] for i in q['indices']],dtype=torch.float64);removed=ys-v[:,None]*interaction.double();recovered=(removed[0]-removed[1]-removed[2]+removed[3])/4;factorerr=max(factorerr,float(recovered.norm()/interaction.double().norm().clamp_min(1e-20)))
    for state,i in enumerate(q['indices']):
     sign=ex[i]['valence'];arms={'identity':torch.zeros_like(interaction),'remove':-sign*interaction,'random_plus':sign*noise,'random_minus':-sign*noise}
     for arm,edit in arms.items():buckets[len(ex[i]['ids'])].append(dict(q=q,state=state,i=i,arm=arm,edit=edit))
   for length,jj in sorted(buckets.items()):
    for st in range(0,len(jj),24):
     batch=jj[st:st+24];ids=torch.tensor([ex[j['i']]['ids'] for j in batch],device='cuda');rep=torch.stack([cap[j['i']][layer,kind]+j['edit'] for j in batch]);y,_=fn(model,ids,patch=(layer,rep));actual+=len(batch)
     for k,j in enumerate(batch):
      out=y[k].double().cpu();base=native[j['i']];lp=out.log_softmax(0);ln=base.log_softmax(0);margin=measure(out,ex[j['i']]['groups'])['margin'];bm=measure(base,ex[j['i']]['groups'])['margin'];val=ex[j['i']]['valence']
      if j['arm']=='identity':ident=max(ident,float((out-base).abs().max()))
      records.append({key:j['q'][key] for key in ['id','format','source','cohort']}|dict(layer=layer,kind=kind,state=j['state'],arm=j['arm'],margin=margin,native_margin=bm,loss=val*(bm-margin),KL=float((ln.exp()*(ln-lp)).sum()),binary_correct=val*margin>0,edit_norm=float(j['edit'].norm())))
   if layer in [0,8,17]:
    q=qs[0];i=q['indices'][0];rep=(cap[i][layer,kind]-co[q['id']][layer,kind][3].float()).unsqueeze(0);ids=torch.tensor([ex[i]['ids']],device='cuda');a,_=fn(model,ids,patch=(layer,rep));b=manual(model,ids,layer,kind,rep);actual+=2;manualerr=max(manualerr,float((a-b).abs().max()))
   print('DONE',kind,layer,len(records),flush=True)
 grouped=defaultdict(list)
 for r in records:grouped[r['format'],r['cohort'],r['layer'],r['kind'],r['arm']].append(r)
 summary=[]
 for key,rr in grouped.items():summary.append(dict(zip(['format','cohort','layer','kind','arm'],key))|dict(n=len(rr),mean_loss=float(np.mean([r['loss'] for r in rr])),rms_loss=float(np.mean([r['loss']**2 for r in rr])**.5),mean_KL=float(np.mean([r['KL'] for r in rr])),binary_correct=float(np.mean([r['binary_correct'] for r in rr]))))
 ranking=[]
 for kind in ['attention','mlp']:
  for layer in range(18):
   ss={r['arm']:r for r in summary if r['format']=='A' and r['cohort']=='familiar_adjectives' and r['layer']==layer and r['kind']==kind};r=ss['remove'];ranking.append(dict(layer=layer,kind=kind,mean_loss=r['mean_loss'],mean_KL=r['mean_KL'],binary_correct=r['binary_correct'],random_rms=((ss['random_plus']['rms_loss']**2+ss['random_minus']['rms_loss']**2)/2)**.5))
 ranking.sort(key=lambda r:r['mean_loss'],reverse=True);audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),whole_model_forwards=actual,expected_forwards=len(ex)*36*4+12,identity_logits_max_abs=ident,random_norm_relative=normerr,factorial_removal_relative=factorerr,independent_manual_patch_max_abs=manualerr,finite=all(np.isfinite(r['KL']) for r in records),no_training=True,test_evaluations=0)
 assert audits['hashes_unchanged'] and actual==audits['expected_forwards'] and ident<1e-4 and max(normerr,factorerr)<1e-5 and manualerr<1e-4 and audits['finite']
 (ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,competence=panel['competence'],records=records,summary=summary,ranking=ranking,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(ranking,audits,flush=True)
if __name__=='__main__':main()

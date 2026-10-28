"""H968 exact final-token interaction accounting through residual writes."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward
from pathlib import Path
import torch,json,numpy as np,time
S='bilin18_valence_interaction_trace'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'captures':ART/'bilin18_factorial_valence_negation_captures.pt','panel':ART/'bilin18_factorial_valence_negation.json','model':SNAPSHOT/'pytorch_model.bin','forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py'};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['captures'],weights_only=False);ex=data['examples'];quartets=data['quartets'];native=data['native'];panel=json.loads(paths['panel'].read_text())
 cfg={'experiment':'H968','hashes':hashes,'design':'Capture all18 layers: block input, x0, attention output, normalized MLP input, MLP output, block output at last token. Exact factorial coefficients and learned-scale residual recurrence. Trace contributions to raw pre-MLP12 interaction by multiplying each earlier attention/MLP write by subsequent residual scales.','limits':'Frozen native trajectory additive attribution, not intervention mediation; downstream writes are held at their observed values for bookkeeping and would change under interventions. Attention writes can transport interactions formed at earlier token positions. Final-token alignment is functional, not equal absolute position. Raw normalized-input interactions include normalization effects.','selection':'Rank signed projection contribution to raw pre-MLP12 interaction on familiar original-format competent quartets. Other cohorts diagnostic; no tuning on original reserved test.','next':'Use dominant upstream attention/MLP site for finite intervention and token-position tracing, rather than interpreting output norm alone as importance.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 buf={};hooks=[]
 def pre(l):
  def h(module,args):
   x,bus,x0=args;buf[l,'input']=x[:,-1].detach().cpu();buf[l,'x0']=x0[:,-1].detach().cpu()
  return h
 def capture(l,key,tupleout=False):
  def h(module,args,out):buf[l,key]=(out[0] if tupleout else out)[:,-1].detach().cpu()
  return h
 def mp(l):
  def h(module,args):buf[l,'normalized']=args[0][:,-1].detach().cpu()
  return h
 for l,b in enumerate(model.blocks):hooks.extend([b.register_forward_pre_hook(pre(l)),b.attn.register_forward_hook(capture(l,'attention',True)),b.mlp.register_forward_pre_hook(mp(l)),b.mlp.register_forward_hook(capture(l,'mlp')),b.register_forward_hook(capture(l,'output',True))])
 captures={};actual=0;replay=0.
 try:
  for length in sorted({len(e['ids']) for e in ex}):
   ii=[i for i,e in enumerate(ex) if len(e['ids'])==length]
   for st in range(0,len(ii),24):
    inds=ii[st:st+24];y,_=forward(model,torch.tensor([ex[i]['ids'] for i in inds],device='cuda'));actual+=len(inds)
    for k,i in enumerate(inds):captures[i]={key:val[k].clone() for key,val in buf.items()};replay=max(replay,float((y[k].double().cpu()-native[i]).abs().max()))
 finally:
  for h in hooks:h.remove()
 scales=[b.lambdas.detach().double().cpu().tolist() for b in model.blocks];H=torch.tensor([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],dtype=torch.float64);records=[];origins=[];recurrence=0.;normerr=0.;algebra=0.;traceerr=0.;stable_algebra=0.;double_algebra=0.;allcoeff={}
 for q in quartets:
  coeff={}
  for l in range(18):
   for key in ['input','x0','attention','normalized','mlp','output']:coeff[l,key]=H.T@torch.stack([captures[i][l,key] for i in q['indices']]).double()/4
   raw=torch.stack([scales[l][0]*captures[i][l,'input']+scales[l][1]*captures[i][l,'x0']+captures[i][l,'attention'] for i in q['indices']]);coeff[l,'raw']=H.T@raw.double()/4
   rr=coeff[l,'input']*scales[l][0]+coeff[l,'x0']*scales[l][1]+coeff[l,'attention']+coeff[l,'mlp'];recurrence=max(recurrence,float((rr-coeff[l,'output']).norm()/coeff[l,'output'].norm()))
   normerr=max(normerr,float((torch.nn.functional.rms_norm(raw,(1152,))-torch.stack([captures[i][l,'normalized'] for i in q['indices']])).abs().max()))
  allcoeff[q['id']]=coeff
 # Batch all quartet coefficient rows per layer for exact MLP interaction decomposition.
 for l,b in enumerate(model.blocks):
  assert not b.mlp.config.gated
  L=b.mlp.Left.weight.detach().double();R=b.mlp.Right.weight.detach().double();D=b.mlp.Down.weight.detach().double()
  def B(u,v):return ((u@L.T)*(v@R.T))@D.T
  C=torch.stack([allcoeff[q['id']][l,'normalized'] for q in quartets]).cuda();inherit=B(C[:,0],C[:,3])+B(C[:,3],C[:,0]);local=B(C[:,1],C[:,2])+B(C[:,2],C[:,1]);tot=inherit+local
  X=torch.einsum('sc,qcd->qsd',H.cuda(),C);Y=B(X,X)+b.mlp.Down_bias.detach().double();exact=(Y[:,0]-Y[:,1]-Y[:,2]+Y[:,3])/4
  double_algebra=max(double_algebra,float((tot-exact).norm()/exact.norm()))
  for k,q in enumerate(quartets):
   c=allcoeff[q['id']];observed=c[l,'mlp'][3].cuda();stable_algebra=max(stable_algebra,float((tot[k]-observed).norm()/c[l,'mlp'].norm().clamp_min(1e-12)));algebra=max(algebra,float((tot[k]-observed).norm()/observed.norm().clamp_min(1e-12)));r={key:q[key] for key in ['id','format','cohort','source']};r.update(layer=l,interaction_norms={key:float(c[l,key][3].norm()) for key in ['input','attention','raw','normalized','mlp','output']},local_norm=float(local[k].norm()),inherited_norm=float(inherit[k].norm()),local_dot_total=float(local[k]@tot[k]),inherited_dot_total=float(inherit[k]@tot[k]),mlp_total_squared_norm=float(tot[k].square().sum()));records.append(r)
 for q in quartets:
  c=allcoeff[q['id']];target=c[12,'raw'][3];pieces=[]
  for l in range(13):
   scale=float(np.prod([scales[t][0] for t in range(l+1,13)]))
   for kind in ['attention','mlp']:
    if l==12 and kind=='mlp':continue
    p=c[l,kind][3]*scale;pieces.append(p);origins.append({k:q[k] for k in ['id','format','cohort','source']}|dict(layer=l,kind=kind,scale=scale,dot_target=float(p@target),target_squared_norm=float(target.square().sum()),contribution_norm=float(p.norm())))
  # x0 and block0-input have zero interaction: same final token per quartet.
  assert c[0,'input'][3].abs().max()==0 and all(c[l,'x0'][3].abs().max()==0 for l in range(13))
  traceerr=max(traceerr,float((sum(pieces)-target).norm()/target.norm()))
 summary=[];origin_summary=[]
 for fmt in ['A','B','C']:
  for cohort in ['familiar_adjectives','new_to_H965_adjectives']:
   for l in range(18):
    rr=[r for r in records if r['format']==fmt and r['cohort']==cohort and r['layer']==l];den=sum(r['mlp_total_squared_norm'] for r in rr);summary.append(dict(format=fmt,cohort=cohort,layer=l,mean_interaction_norms={key:float(np.mean([r['interaction_norms'][key] for r in rr])) for key in rr[0]['interaction_norms']},local_vector_share=sum(r['local_dot_total'] for r in rr)/den))
   for l in range(13):
    for kind in ['attention','mlp']:
     rr=[r for r in origins if r['format']==fmt and r['cohort']==cohort and r['layer']==l and r['kind']==kind]
     if rr:origin_summary.append(dict(format=fmt,cohort=cohort,layer=l,kind=kind,signed_projection_share=sum(r['dot_target'] for r in rr)/sum(r['target_squared_norm'] for r in rr)))
 rank=sorted([r for r in origin_summary if r['format']=='A' and r['cohort']=='familiar_adjectives'],key=lambda r:r['signed_projection_share'],reverse=True)
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),whole_model_forwards=actual,expected_forwards=len(ex),capture_logits_max_abs=replay,residual_recurrence_relative=recurrence,normalization_replay_max_abs=normerr,mlp_interaction_replay_relative=algebra,mlp_interaction_error_relative_to_all_coefficients=stable_algebra,double_precision_interaction_identity=double_algebra,prior_failed_attempt_forwards=192,total_forwards_including_audit_retries=actual+192,raw_MLP12_trace_relative=traceerr,zero_initial_interaction=True,finite=all(np.isfinite(r['signed_projection_share']) for r in origin_summary),no_training=True,test_evaluations=0)
 print('AUDITS',audits,flush=True)
 torch.save(dict(captures=captures,coefficients=allcoeff,scales=scales),ART/f'{S}_captures.pt')
 (ART/f'{S}_audit_diagnostic.json').write_text(json.dumps(audits,indent=2)+'\n')
 assert audits['hashes_unchanged'] and actual==len(ex) and replay<1e-4 and recurrence<1e-5 and normerr<1e-5 and stable_algebra<1e-5 and double_algebra<1e-10 and traceerr<1e-5 and audits['finite']
 torch.save(dict(captures=captures,coefficients=allcoeff,scales=scales),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,competence=panel['competence'],records=records,origins=origins,summary=summary,origin_summary=origin_summary,ranking=rank,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(rank,audits,flush=True)
if __name__=='__main__':main()

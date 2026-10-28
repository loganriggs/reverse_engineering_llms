"""H972 source-position factorial trace before attention5."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward
from pathlib import Path
import torch,json,numpy as np,time
S='bilin18_source_position_composition'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'data':ART/'bilin18_factorial_valence_negation_captures.pt','panel':ART/'bilin18_factorial_valence_negation.json','sources':ART/'bilin18_attention5_sources_captures.pt','model':SNAPSHOT/'pytorch_model.bin','forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py'};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);sources=torch.load(paths['sources'],weights_only=False);panel=json.loads(paths['panel'].read_text());ex=data['examples'];qs=data['quartets'];native=data['native'];positions={i:{'adjective':max(t for t,s in enumerate(sources['slots'][i]) if s.startswith('adjective:')),'period':sources['slots'][i].index('suffix:0')} for i in range(len(ex))}
 cfg={'experiment':'H972','hashes':hashes,'design':'Capture adjective-last-subtoken and following-period positions at blocks0-5: input, embedding skip, attention input/output, normalized MLP input/output, residual output. Exact factorial bilinear decomposition and residual accounting. Separately capture attention5 current-value and native layer0 value bus.','alignment':'Use last adjective subtoken; competent familiar adjectives are all single tokens. Multi-subtoken failed-cohort diagnostics do not represent the sum of all adjective-source contributions. Period verified by token ID13.','limits':'Descriptive native trajectory only; local versus inherited refers to normalized MLP input, so normalization can introduce interaction. No claim that earliest nonzero term is semantic. Failed cohorts remain diagnostics; no fitting or reserved-test access.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');assert all(e['ids'][positions[i]['period']]==13 for i,e in enumerate(ex))
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 buf={};batchindices=[]
 def take(x):
  return torch.stack([torch.stack([x[k,positions[i][role]] for role in ['adjective','period']]) for k,i in enumerate(batchindices)]).detach().cpu()
 def pre(l):
  def h(module,args):
   x,bus,x0=args;buf[l,'input']=take(x);buf[l,'x0']=take(x0)
  return h
 def out(l,key,istuple=False):
  def h(module,args,y):buf[l,key]=take(y[0] if istuple else y)
  return h
 def mp(l):
  def h(module,args):buf[l,'normalized']=take(args[0])
  return h
 def ap(l):
  def h(module,args):
   x,bus=args;buf[l,'attn_input']=take(x)
   if l==5:
    b,t,c=x.shape;current=module.c_v(x).reshape(b,t,module.n_head,module.head_dim);buf[l,'current_value']=take(current);buf[l,'value_bus']=take(bus.view_as(current))
  return h
 hooks=[]
 for l in range(6):
  b=model.blocks[l];hooks.extend([b.register_forward_pre_hook(pre(l)),b.attn.register_forward_pre_hook(ap(l)),b.attn.register_forward_hook(out(l,'attention',True)),b.mlp.register_forward_pre_hook(mp(l)),b.mlp.register_forward_hook(out(l,'mlp')),b.register_forward_hook(out(l,'output',True))])
 captures={};actual=0;replay=0.;mixerr=0.
 try:
  for length in sorted({len(e['ids']) for e in ex}):
   ii=[i for i,e in enumerate(ex) if len(e['ids'])==length]
   for st in range(0,len(ii),24):
    batchindices=ii[st:st+24];y,_=forward(model,torch.tensor([ex[i]['ids'] for i in batchindices],device='cuda'));actual+=len(batchindices)
    for k,i in enumerate(batchindices):
     captures[i]={key:v[k].clone() for key,v in buf.items()};replay=max(replay,float((y[k].double().cpu()-native[i]).abs().max()));mix=(1-model.blocks[5].attn.lamb.cpu())*captures[i][5,'current_value']+model.blocks[5].attn.lamb.cpu()*captures[i][5,'value_bus'];old=torch.stack([sources['captures'][i]['V'][positions[i][r]] for r in ['adjective','period']]);mixerr=max(mixerr,float((mix-old).norm()/old.norm()))
 finally:
  for h in hooks:h.remove()
 H=torch.tensor([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],dtype=torch.float64);scales=[b.lambdas.detach().double().cpu().tolist() for b in model.blocks];coeff={};records=[];value_records=[];recerr=0.;mlperr=0.;initial=0.;parts={}
 for q in qs:
  for ri,role in enumerate(['adjective','period']):
   c={}
   for key in captures[q['indices'][0]]:c[key]=torch.einsum('sc,s...->c...',H,torch.stack([captures[i][key][ri] for i in q['indices']]).double())/4
   coeff[q['id'],role]=c;initial=max(initial,float(c[0,'input'][3].abs().max()))
   for l in range(6):
    pred=scales[l][0]*c[l,'input']+scales[l][1]*c[l,'x0']+c[l,'attention']+c[l,'mlp'];recerr=max(recerr,float((pred-c[l,'output']).norm()/c[l,'output'].norm()))
 for l in range(6):
  m=model.blocks[l].mlp;L=m.Left.weight.detach().double();R=m.Right.weight.detach().double();D=m.Down.weight.detach().double()
  def B(u,v):return ((u@L.T)*(v@R.T))@D.T
  for q in qs:
   for role in ['adjective','period']:
    c=coeff[q['id'],role];C=c[l,'normalized'].cuda();inherit=B(C[0],C[3])+B(C[3],C[0]);local=B(C[1],C[2])+B(C[2],C[1]);total=inherit+local;mlperr=max(mlperr,float((total-c[l,'mlp'][3].cuda()).norm()/c[l,'mlp'].norm()));parts[q['id'],role,l]=dict(inherited=inherit.float().cpu(),local=local.float().cpu());r={k:q[k] for k in ['id','format','cohort','source']};r.update(role=role,layer=l,interaction_norms={key:float(c[l,key][3].norm()) for key in ['input','attn_input','attention','normalized','mlp','output']},local_dot_total=float(local@total),total_squared_norm=float(total.square().sum()),local_norm=float(local.norm()),inherited_norm=float(inherit.norm()));records.append(r)
 att=model.blocks[5].attn;lam=float(att.lamb);W=sources['projection'];
 for q in qs:
  for role in ['adjective','period']:
   c=coeff[q['id'],role];cur=(1-lam)*torch.einsum('hd,ohd->o',c[5,'current_value'][3],W);bus=lam*torch.einsum('hd,ohd->o',c[5,'value_bus'][3],W);tot=cur+bus;value_records.append({k:q[k] for k in ['id','format','cohort','source']}|dict(role=role,current_projected_interaction_norm=float(cur.norm()),bus_projected_interaction_norm=float(bus.norm()),total_projected_interaction_norm=float(tot.norm())))
 summary=[]
 for fmt in ['A','B','C']:
  for cohort in ['familiar_adjectives','new_to_H965_adjectives']:
   for role in ['adjective','period']:
    for l in range(6):
     rr=[r for r in records if r['format']==fmt and r['cohort']==cohort and r['role']==role and r['layer']==l];den=sum(r['total_squared_norm'] for r in rr);summary.append(dict(format=fmt,cohort=cohort,role=role,layer=l,mean_interaction_norms={key:float(np.mean([r['interaction_norms'][key] for r in rr])) for key in rr[0]['interaction_norms']},local_vector_share=sum(r['local_dot_total'] for r in rr)/den))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),whole_model_forwards=actual,expected_forwards=len(ex),native_logits_max_abs=replay,value_mix_replay_relative=mixerr,residual_recurrence_relative=recerr,MLP_interaction_error_relative_to_all_coefficients=mlperr,initial_interaction_max_abs=initial,finite=all(np.isfinite(r['local_vector_share']) for r in summary),no_training=True,test_evaluations=0)
 assert audits['hashes_unchanged'] and actual==len(ex) and replay<1e-4 and max(mixerr,recerr,mlperr)<1e-5 and initial==0 and audits['finite']
 torch.save(dict(captures=captures,coefficients=coeff,parts=parts,positions=positions,scales=scales),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,competence=panel['competence'],records=records,value_records=value_records,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print([r for r in summary if r['format']=='A' and r['cohort']=='familiar_adjectives'],audits,flush=True)
if __name__=='__main__':main()

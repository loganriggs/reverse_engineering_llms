"""H979 single-state scalar quadratic readouts and counterfactual interventions."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from pathlib import Path
from collections import defaultdict
import torch,json,numpy as np,time
S='bilin18_single_state_composition'
def predict(X,p):
 z=X-p['center'];out=p['constant']+z@p['linear'];Q=p['Q'];return out+.5*((z@Q)@p['K']*(z@Q)).sum(-1)
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'old':ART/'bilin18_source_position_composition_captures.pt','bases':ART/'bilin18_output_aware_composition_bases.pt','data':ART/'bilin18_prospective_composition_panel_captures.pt','panel':ART/'bilin18_prospective_composition_panel.json','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};old=torch.load(paths['old'],weights_only=False);base=torch.load(paths['bases'],weights_only=False)['bases'];d=torch.load(paths['data'],weights_only=False);panel=json.loads(paths['panel'].read_text());ex=d['examples'];qs=d['quartets'];cap=d['captures'];native=d['native'];model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 m=model.blocks[3].mlp;L=m.Left.weight.detach().double();R=m.Right.weight.detach().double();D=m.Down.weight.detach().double();u=base['O'][:,0].cuda();g=D.T@u;F=L.T@(g[:,None]*R)+R.T@(g[:,None]*L);F=(F+F.T)/2;bias=m.Down_bias.detach().double()@u
 # Reconstruct only discovery normalized states from their four-state coefficients.
 H=torch.tensor([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],dtype=torch.float64);train=torch.cat([H@old['coefficients'][f'A-{i}','adjective'][3,'normalized'] for i in range(4)]).cuda();mu=train.mean(0);grad=F@mu;constant=.5*mu@F@mu+bias
 cfg={'experiment':'H979','hashes':hashes,'design':'Predict scalar u dot MLP3(x) from a single normalized adjective activation. Frozen output direction and discovery negation span. Aware subspace Q spans N and F*N. Center on discovery16 states; retain exact constant and full linear Taylor term, restrict quadratic to Q.T F Q. Compare rank2/rank8 aware spans, discovery PCA8, random8, linear tangent, and uncentered aware8.','inputs':'Predict accepts an independent activation and fixed parameters only; no evaluation quartet factors, labels, teacher outputs or batch-dependent statistics. Labels used for original discovery bases and evaluation grouping only.','interventions':'For every directed pair within each quartet, add u*(predicted_scalar(donor)-predicted_scalar(recipient)) to native recipient adjective-position MLP3 write. Compare with exact scalar-output delta, not full MLP replacement or earlier local-only intervention.','limits':'Only one frozen output direction, not full layer faithfulness. Exact weight knowledge and supervised discovery used. Development evaluation panel already inspected; a separate prospective panel is still required.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 params={}
 def add(name,Q,center=mu,linear=grad,c=constant):params[name]=dict(center=center.cpu(),linear=linear.cpu(),constant=c.cpu(),Q=Q.cpu(),K=(Q.T@F@Q).cpu())
 for rank in [1,4]:
  N=base['N'][:,:rank].cuda();cat=torch.cat([N,F@N],1);U,sv,_=torch.linalg.svd(cat,full_matrices=False);Q=U[:,sv>sv.max()*1e-10];add(f'aware_n{rank}',Q)
  if rank==4:add('uncentered_aware_n4',Q,torch.zeros_like(mu),torch.zeros_like(mu),bias)
 _,_,V=torch.linalg.svd(train-mu,full_matrices=False);add('PCA8',V[:8].T);gen=torch.Generator(device='cuda').manual_seed(979);Q=torch.linalg.qr(torch.randn(1152,8,device='cuda',dtype=torch.float64,generator=gen)).Q;add('random8',Q);add('linear',torch.zeros(1152,0,device='cuda',dtype=torch.float64))
 frozen=ART/f'{S}_models.pt';torch.save(dict(params=params,output_direction=u.cpu(),fit_ids=[f'A-{i}' for i in range(4)]),frozen);frozenhash=sha(frozen);X=torch.stack([cap[i]['normalized'] for i in range(len(ex))]).double();truth=torch.stack([cap[i]['adjective_write'] for i in range(len(ex))]).double()@u.cpu();exact=.5*((X.cuda()@F)*X.cuda()).sum(1)+bias;scalar_replay=float((exact.cpu()-truth).abs().max());preds={name:predict(X,p) for name,p in params.items()};pointwise=max(float((torch.stack([predict(X[i:i+1],p)[0] for i in range(len(X))])-preds[name]).abs().max()) for name,p in params.items());geometry=[]
 for q in qs:
  ii=q['indices']
  for name,y in preds.items():geometry.append({key:q[key] for key in ['id','format','cohort','source']}|dict(mode=name,scalar_rmse=float((y[ii]-truth[ii]).square().mean().sqrt()),scalar_bias=float((y[ii]-truth[ii]).mean()),centered_scalar_error=float(((y[ii]-y[ii].mean())-(truth[ii]-truth[ii].mean())).square().mean().sqrt()),teacher_scalar_rms=float(truth[ii].square().mean().sqrt())))
 refs=[];jobs=[]
 for q in qs:
  for rec,i in enumerate(q['indices']):
   for don,di in enumerate(q['indices']):
    if rec==don:continue
    edge='both' if ex[i]['a']!=ex[di]['a'] and ex[i]['n']!=ex[di]['n'] else ('adjective' if ex[i]['a']!=ex[di]['a'] else 'negation');b=dict(q=q,rec=rec,don=don,i=i,di=di,edge=edge);refs.append(b|dict(mode='teacher',delta=float(truth[di]-truth[i])));jobs.append(b|dict(mode='identity',delta=0.))
    for name,y in preds.items():jobs.append(b|dict(mode=name,delta=float(y[di]-y[i])))
 current=[]
 def patch(module,args,out):
  y=out.clone()
  for k,j in enumerate(current):i=j['i'];y[k,ex[i]['adjective_position']]=cap[i]['adjective_write'].to(y)+j['delta']*u.to(y)
  return y
 hook=model.blocks[3].mlp.register_forward_hook(patch);full={};records=[];actual=0;ident=0.
 def execute(jobs,reference=False):
  nonlocal current,actual,ident
  buckets=defaultdict(list)
  for j in jobs:buckets[len(ex[j['i']]['ids'])].append(j)
  for length,jj in sorted(buckets.items()):
   for st in range(0,len(jj),24):
    current=jj[st:st+24];out,_=forward(model,torch.tensor([ex[j['i']]['ids'] for j in current],device='cuda'));actual+=len(current)
    for k,j in enumerate(current):
     y=out[k].double().cpu();b=native[j['i']];key=j['i'],j['di']
     if reference:full[key]=y
     ref=full[key];lp=y.log_softmax(0);ln=b.log_softmax(0);lr=ref.log_softmax(0);margin=measure(y,ex[j['i']]['groups'])['margin'];bm=measure(b,ex[j['i']]['groups'])['margin'];tm=measure(ref,ex[j['i']]['groups'])['margin']
     if j['mode']=='identity':ident=max(ident,float((y-b).abs().max()))
     records.append({key:j['q'][key] for key in ['id','format','cohort','source']}|dict(recipient=j['rec'],donor=j['don'],edge=j['edge'],mode=j['mode'],margin=margin,margin_delta=margin-bm,teacher_margin_delta=tm-bm,KL=float((lr.exp()*(lr-lp)).sum()),baseline_KL=float((lr.exp()*(lr-ln)).sum())))
 try:execute(refs,True);execute(jobs)
 finally:hook.remove()
 summary=[];grouped=defaultdict(list)
 for r in records:grouped[r['format'],r['cohort'],r['edge'],r['mode']].append(r);grouped[r['format'],r['cohort'],'all',r['mode']].append(r)
 for key,rr in grouped.items():summary.append(dict(zip(['format','cohort','edge','mode'],key))|dict(n=len(rr),KL_ratio=sum(r['KL'] for r in rr)/sum(r['baseline_KL'] for r in rr),margin_error_rms=float(np.mean([(r['margin_delta']-r['teacher_margin_delta'])**2 for r in rr])**.5)))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),frozen_models_unchanged=sha(frozen)==frozenhash,whole_model_forwards=actual,expected_forwards=len(refs)+len(jobs),identity_logits_max_abs=ident,scalar_weight_replay_max_abs=scalar_replay,pointwise_vs_batch_prediction_max_abs=pointwise,finite=all(np.isfinite(r['KL']) for r in records),no_gradient_training=True,test_evaluations=0)
 assert audits['hashes_unchanged'] and audits['frozen_models_unchanged'] and actual==len(refs)+len(jobs) and ident<1e-4 and scalar_replay<.1 and pointwise<1e-8 and audits['finite']
 torch.save(dict(predictions=preds,teacher=truth),ART/f'{S}_predictions.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,competence=panel['summary'],geometry=geometry,records=records,summary=summary,model_dimensions={k:p['Q'].shape[1] for k,p in params.items()},audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

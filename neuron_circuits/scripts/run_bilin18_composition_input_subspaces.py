"""H977 frozen discovery input spans for local bilinear composition."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from pathlib import Path
from collections import defaultdict
import torch,json,numpy as np,time
S='bilin18_composition_input_subspaces'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'discovery':ART/'bilin18_source_position_composition_captures.pt','data':ART/'bilin18_prospective_composition_panel_captures.pt','panel':ART/'bilin18_prospective_composition_panel.json','teacher_parts':ART/'bilin18_prospective_route_transfer_parts.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};old=torch.load(paths['discovery'],weights_only=False);d=torch.load(paths['data'],weights_only=False);panel=json.loads(paths['panel'].read_text());teacher=torch.load(paths['teacher_parts'],weights_only=False)['parts'];ex=d['examples'];qs=d['quartets'];cap=d['captures'];native=d['native'];fitids=[f'A-{i}' for i in range(4)]
 cfg={'experiment':'H977','hashes':hashes,'fit_ids':fitids,'design':'SVD adjective and negation main-factor vectors at normalized adjective MLP3 input from original four quartets only. Freeze rank1 and rank4 spans. Project new quartet factor vectors, compute exact MLP bilinear cross term from projections, subtract a*n predicted term and measure fidelity to teacher local-term removal.','comparators':'Adjective-only and negation-only rank4 restrictions; output rank1/rank4 projection of teacher local term using discovery output spans; three seeded random rank4 input-span pairs; identity. No norm rescaling.','limitations':'New factors still require four teacher states. This is span adequacy and operator compression, not state-based encoder generalization. Output-only controls use teacher output coefficients and do not predict computation. Random spans are dimension matched but not data covariance matched.','selection':'No site or rank tuning on evaluation results. Rank4 is maximum discovery span, rank1 first principal direction; rows not normalized before SVD. All cohorts evaluated, native competence gates inherited.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 A=torch.stack([old['coefficients'][qid,'adjective'][3,'normalized'][1] for qid in fitids]);N=torch.stack([old['coefficients'][qid,'adjective'][3,'normalized'][2] for qid in fitids]);O=torch.stack([old['parts'][qid,'adjective',3]['local'].double() for qid in fitids]);_,sa,va=torch.linalg.svd(A,full_matrices=False);_,sn,vn=torch.linalg.svd(N,full_matrices=False);_,so,vo=torch.linalg.svd(O,full_matrices=False);bases={'A':va.T,'N':vn.T,'O':vo.T};random={}
 for seed in range(3):
  gen=torch.Generator().manual_seed(977000+seed);random[seed]=[torch.linalg.qr(torch.randn(1152,4,generator=gen,dtype=torch.float64)).Q for _ in range(2)]
 frozen=ART/f'{S}_bases.pt';torch.save(dict(bases=bases,random=random,fitids=fitids,singular_values=dict(A=sa,N=sn,O=so)),frozen);frozenhash=sha(frozen);model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 m=model.blocks[3].mlp;L=m.Left.weight.detach().double();R=m.Right.weight.detach().double();D=m.Down.weight.detach().double()
 def B(u,v):return ((u@L.T)*(v@R.T))@D.T
 def cross(a,n):return B(a,n)+B(n,a)
 def project(x,U):U=U.to(x);return (x@U)@U.T
 H=torch.tensor([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],dtype=torch.float64,device='cuda');predictions={};geometry=[];teachererr=0.;contractionerr=0.
 # Frozen compact operator maps four adjective and four negation coordinates to an output.
 Ua=bases['A'].cuda();Un=bases['N'].cuda();T=torch.stack([torch.stack([cross(Ua[:,i],Un[:,j]) for j in range(4)]) for i in range(4)]);torch.save(T.cpu(),ART/f'{S}_operator.pt')
 for q in qs:
  C=H.T@torch.stack([cap[i]['normalized'] for i in q['indices']]).double().cuda()/4;a,n=C[1],C[2];truth=cross(a,n);teachererr=max(teachererr,float((truth-teacher[q['id']]['local'].double().cuda()).norm()/truth.norm()));aa=project(a,bases['A']);nn=project(n,bases['N']);p={'input_rank1':cross(project(a,bases['A'][:,:1]),project(n,bases['N'][:,:1])),'input_rank4':cross(aa,nn),'adjective_only4':cross(aa,n),'negation_only4':cross(a,nn),'output_rank1':project(truth,bases['O'][:,:1]),'output_rank4':project(truth,bases['O'])}
  compact=torch.einsum('i,j,ijo->o',a@Ua,n@Un,T);contractionerr=max(contractionerr,float((compact-p['input_rank4']).norm()/p['input_rank4'].norm()))
  for seed,(ra,rn) in random.items():p[f'random_input4_{seed}']=cross(project(a,ra),project(n,rn))
  predictions[q['id']]={key:v.float().cpu() for key,v in p.items()};predictions[q['id']]['teacher']=truth.float().cpu();predictions[q['id']]['identity']=torch.zeros(1152)
  for name,y in p.items():geometry.append({key:q[key] for key in ['id','format','cohort','source']}|dict(mode=name,relative_error=float((y-truth).norm()/truth.norm()),cosine=float((y@truth)/(y.norm()*truth.norm()).clamp_min(1e-20)),a_energy_retained=float(aa.square().sum()/a.square().sum()),n_energy_retained=float(nn.square().sum()/n.square().sum())))
 refs=[];jobs=[]
 for q in qs:
  for state,i in enumerate(q['indices']):
   for name,p in predictions[q['id']].items():
    j=dict(q=q,state=state,i=i,mode=name,edit=-ex[i]['valence']*p);(refs if name=='teacher' else jobs).append(j)
 current=[]
 def patch(module,args,out):
  y=out.clone()
  for k,j in enumerate(current):i=j['i'];y[k,ex[i]['adjective_position']]=cap[i]['adjective_write'].to(y)+j['edit'].to(y)
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
     y=out[k].double().cpu();base=native[j['i']]
     if reference:full[j['i']]=y
     ref=full[j['i']];lp=y.log_softmax(0);ln=base.log_softmax(0);lr=ref.log_softmax(0);margin=measure(y,ex[j['i']]['groups'])['margin'];bm=measure(base,ex[j['i']]['groups'])['margin'];tm=measure(ref,ex[j['i']]['groups'])['margin'];v=ex[j['i']]['valence']
     if j['mode']=='identity':ident=max(ident,float((y-base).abs().max()))
     records.append({key:j['q'][key] for key in ['id','format','cohort','source']}|dict(state=j['state'],mode=j['mode'],margin=margin,loss=v*(bm-margin),teacher_loss=v*(bm-tm),KL=float((lr.exp()*(lr-lp)).sum()),baseline_KL=float((lr.exp()*(lr-ln)).sum())))
 try:execute(refs,True);execute(jobs)
 finally:hook.remove()
 summary=[];grouped=defaultdict(list)
 for r in records:grouped[r['format'],r['cohort'],r['mode']].append(r)
 for key,rr in grouped.items():summary.append(dict(zip(['format','cohort','mode'],key))|dict(n=len(rr),mean_loss=float(np.mean([r['loss'] for r in rr])),teacher_loss=float(np.mean([r['teacher_loss'] for r in rr])),KL_ratio=sum(r['KL'] for r in rr)/sum(r['baseline_KL'] for r in rr),margin_error_rms=float(np.mean([(r['loss']-r['teacher_loss'])**2 for r in rr])**.5)))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),frozen_bases_unchanged=sha(frozen)==frozenhash,whole_model_forwards=actual,expected_forwards=len(refs)+len(jobs),identity_logits_max_abs=ident,teacher_local_replay_relative=teachererr,compact_operator_contraction_relative=contractionerr,orthonormality_max_abs=max(float((u.T@u-torch.eye(4)).abs().max()) for u in bases.values()),finite=all(np.isfinite(r['KL']) for r in records),no_training=True,test_evaluations=0)
 assert audits['hashes_unchanged'] and audits['frozen_bases_unchanged'] and actual==len(refs)+len(jobs) and ident<1e-4 and teachererr<1e-6 and contractionerr<1e-10 and audits['finite']
 torch.save(dict(predictions=predictions),ART/f'{S}_predictions.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,competence=panel['summary'],geometry=geometry,records=records,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

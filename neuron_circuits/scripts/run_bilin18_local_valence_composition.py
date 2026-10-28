"""H967: exact bilinear attribution of factorial interaction at MLP12."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from pathlib import Path
from collections import defaultdict
import torch,json,numpy as np,time
S='bilin18_local_valence_composition'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'runner':Path(__file__),'captures':ART/'bilin18_factorial_valence_negation_captures.pt','panel':ART/'bilin18_factorial_valence_negation.json','directions':ART/'bilin18_shared_task_directions_directions.pt','model':SNAPSHOT/'pytorch_model.bin','forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py'};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['captures'],weights_only=False);panel=json.loads(paths['panel'].read_text());ex=data['examples'];quartets=data['quartets'];native=data['native'];writes=data['writes'];dirs=torch.load(paths['directions'],weights_only=False)['directions']
 cfg={'experiment':'H967','hashes':hashes,'design':'Capture actual normalized MLP12 inputs and raw pre-normalization residuals on locked H966 panel. Let x=c0+a*ca+n*cn+a*n*ci after RMS normalization. Define B(u,v)=Down((Left u)*(Right v)). Exact output interaction is inherited=B(c0,ci)+B(ci,c0), local=B(ca,cn)+B(cn,ca). Bias contributes only constant.','interventions':'For each native state remove a*n times inherited interaction, local interaction, or both from its MLP12 output. Identity control. This algebraic output intervention may be off-manifold; it is not an input intervention or proof of necessity under natural edits.','normalization':'Inherited refers to interaction already in normalized input. Raw input factorial energy is also reported, but normalization itself can generate interaction. No claim of unique causal attribution to earlier layers before normalization.','metrics':'Full output reconstruction including Down_bias, interaction reconstruction, native-reference KL and margins after component removal. Signed vector projection shares need not be positive or bounded by one.','scope':'No fitting; retain all H966 cohorts, distinguish native-gate failures. No held-out training tasks or original reserved test. Four familiar quartets are the only native-gated cohort.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 block=model.blocks[12];mlp=block.mlp;assert not mlp.config.gated
 captures={};buf={}
 def bh(module,args):
  x,bus,x0=args;buf['raw_start']=(module.lambdas[0]*x[:,-1]+module.lambdas[1]*x0[:,-1]).detach().cpu()
 def ah(module,args,out):buf['raw']=buf['raw_start']+out[0][:,-1].detach().cpu()
 def mh(module,args):buf['normalized']=args[0][:,-1].detach().cpu()
 hooks=[block.register_forward_pre_hook(bh),block.attn.register_forward_hook(ah),mlp.register_forward_pre_hook(mh)];actual=0;caperr=0.;normerr=0.
 try:
  for length in sorted({len(e['ids']) for e in ex}):
   ii=[i for i,e in enumerate(ex) if len(e['ids'])==length]
   for st in range(0,len(ii),24):
    inds=ii[st:st+24];y,_=forward(model,torch.tensor([ex[i]['ids'] for i in inds],device='cuda'));actual+=len(inds)
    for k,i in enumerate(inds):
     captures[i]={name:buf[name][k].clone() for name in ['raw','normalized']};caperr=max(caperr,float((y[k].double().cpu()-native[i]).abs().max()));r=torch.nn.functional.rms_norm(captures[i]['raw'],(1152,));normerr=max(normerr,float((r-captures[i]['normalized']).abs().max()))
 finally:
  for h in hooks:h.remove()
 L=mlp.Left.weight.detach().double();R=mlp.Right.weight.detach().double();D=mlp.Down.weight.detach().double();bias=mlp.Down_bias.detach().double()
 def B(u,v):return ((u@L.T)*(v@R.T))@D.T
 H=torch.tensor([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],dtype=torch.float64,device='cuda')
 # Independent scalar 1D examples check both cross terms and bias behavior.
 toy=torch.tensor([[2.,3.,5.,7.]],dtype=torch.float64,device='cuda').T;tx=H@toy;ty=2*tx*3*tx+11;tc=H.T@ty/4;assert abs(float(tc[3])-12*(2*7+3*5))<1e-10
 factors=[];parts={};writeerr=0.;intererr=0.;exacterr=0.
 for q in quartets:
  X=torch.stack([captures[i]['normalized'] for i in q['indices']]).double().cuda();raw=torch.stack([captures[i]['raw'] for i in q['indices']]).double().cuda();C=H.T@X/4;Cr=H.T@raw/4;Y=B(X,X)+bias;old=torch.stack([writes[i] for i in q['indices']]).double().cuda();writeerr=max(writeerr,float((Y-old).norm()/old.norm()));cy=H.T@Y/4
  inherited=B(C[0],C[3])+B(C[3],C[0]);local=B(C[1],C[2])+B(C[2],C[1]);total=inherited+local;intererr=max(intererr,float((total-cy[3]).norm()/cy[3].norm()));exacterr=max(exacterr,float((H@(H.T@Y/4)-Y).abs().max()));parts[q['id']]=dict(inherited=inherited.float().cpu(),local=local.float().cpu(),total=total.float().cpu())
  en=C[1:].square().sum(1);er=Cr[1:].square().sum(1);tt=float(total.square().sum());row={k:q[k] for k in ['id','format','source','cohort']};row.update(raw_factor_energy=(er/er.sum()).tolist(),normalized_factor_energy=(en/en.sum()).tolist(),total_norm=float(total.norm()),inherited_norm=float(inherited.norm()),local_norm=float(local.norm()),inherited_dot_total=float(inherited@total),local_dot_total=float(local@total),total_squared_norm=tt,inherited_local_cos=float((inherited@local)/(inherited.norm()*local.norm())))
  row['direction_coordinates']={name:{key:float(val@dirs[name].double().cuda()) for key,val in [('inherited',inherited),('local',local),('total',total)]} for name in ['sentiment','negation']};factors.append(row)
 # Remove only the output interaction term; do not alter native residual or attention bus.
 jobs=[]
 for q in quartets:
  for state,i in enumerate(q['indices']):
   for arm in ['identity','remove_inherited','remove_local','remove_total']:
    edit=torch.zeros(1152) if arm=='identity' else -ex[i]['valence']*parts[q['id']][arm.removeprefix('remove_')];jobs.append(dict(q=q,i=i,state=state,arm=arm,edit=edit))
 buckets=defaultdict(list)
 for j in jobs:buckets[len(ex[j['i']]['ids'])].append(j)
 records=[];ident=0.
 for length,jj in sorted(buckets.items()):
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];out,_=forward(model,torch.tensor([ex[j['i']]['ids'] for j in batch],device='cuda'),patch=(12,torch.stack([writes[j['i']]+j['edit'] for j in batch])));actual+=len(batch)
   for k,j in enumerate(batch):
    y=out[k].double().cpu();base=native[j['i']];lp=y.log_softmax(0);ln=base.log_softmax(0);m=measure(y,ex[j['i']]['groups']);bm=measure(base,ex[j['i']]['groups']);v=ex[j['i']]['valence']
    if j['arm']=='identity':ident=max(ident,float((y-base).abs().max()))
    records.append({key:j['q'][key] for key in ['id','format','source','cohort']}|dict(state=j['state'],arm=j['arm'],valence=v,margin=m['margin'],native_margin=bm['margin'],aligned_margin_loss=v*(bm['margin']-m['margin']),KL=float((ln.exp()*(ln-lp)).sum()),binary_correct=v*m['margin']>0,edit_norm=float(j['edit'].norm())))
 summary=[]
 for fmt in ['A','B','C']:
  for cohort in ['familiar_adjectives','new_to_H965_adjectives']:
   ff=[f for f in factors if f['format']==fmt and f['cohort']==cohort];den=sum(f['total_squared_norm'] for f in ff);attribution=dict(inherited_vector_share=sum(f['inherited_dot_total'] for f in ff)/den,local_vector_share=sum(f['local_dot_total'] for f in ff)/den,raw_interaction_energy=float(np.mean([f['raw_factor_energy'][2] for f in ff])),normalized_interaction_energy=float(np.mean([f['normalized_factor_energy'][2] for f in ff])),sentiment_local_coordinate_share=sum(f['direction_coordinates']['sentiment']['local'] for f in ff)/sum(f['direction_coordinates']['sentiment']['total'] for f in ff))
   for arm in ['identity','remove_inherited','remove_local','remove_total']:
    rr=[r for r in records if r['format']==fmt and r['cohort']==cohort and r['arm']==arm];summary.append(dict(format=fmt,cohort=cohort,arm=arm,n=len(rr),mean_aligned_margin_loss=float(np.mean([r['aligned_margin_loss'] for r in rr])),mean_KL=float(np.mean([r['KL'] for r in rr])),binary_correct=float(np.mean([r['binary_correct'] for r in rr])),attribution=attribution))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),whole_model_forwards=actual,expected_forwards=len(ex)+len(jobs),capture_logits_max_abs=caperr,identity_logits_max_abs=ident,normalization_replay_max_abs=normerr,write_replay_relative=writeerr,interaction_algebra_relative=intererr,four_state_reconstruction_max_abs=exacterr,scalar_polynomial_test=True,finite=all(np.isfinite(r['KL']) for r in records),no_training=True,test_evaluations=0)
 assert audits['hashes_unchanged'] and actual==audits['expected_forwards'] and max(caperr,ident)<1e-4 and normerr<1e-5 and writeerr<1e-5 and intererr<1e-10 and audits['finite']
 torch.save(dict(captures=captures,parts=parts),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,competence=panel['competence'],factors=factors,records=records,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(summary,audits,flush=True)
if __name__=='__main__':main()

"""M59 native source-group holdout targets, fixed before outcomes."""
from pathlib import Path
import json,hashlib
import torch
from inspect_model import load_model,trace
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def margin(z,g,v):return float(torch.logsumexp(z.double()[g[v]],0)-torch.logsumexp(z.double()[g[1-v]],0))
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);pp=V/'source_holdout_native_protocol.json';q=json.loads(pp.read_text());op=V/'source_holdout_native.json';assert not op.exists()
 for n,h in q['input_hashes'].items():assert sha(V/n)==h
 model,receipt=load_model();records=[];samples=dict(baselines={},changed={},deltas={});calls=0;energychecks=[]
 for pi,pair in enumerate(q['pairs']):
  caps=[]
  for ep in [0,1]:
   ids=torch.tensor([pair['token_ids'][ep]],device='cuda');out,cap=trace(model,ids);calls+=1;z=out[0,-1];key=pair['id']+f'|{ep}';samples['baselines'][key]=dict(logits=z.cpu(),groups=pair['answer_groups'][ep],variant=pair['variants'][ep]);writes={l:model.transformer.h[l].mlp.Down(cap[l]['products']) for l in q['layers']};caps.append(dict(ids=ids,logits=z,writes=writes,key=key))
  for spec in [s for s in q['assignments'] if s['pair_id']==pair['id']]:
   ep=spec['endpoint'];layer=spec['layer'];region=spec['region'];key=caps[ep]['key'];r=dict(**spec,family=pair['family'],sample_key=key+'|'+str(layer)+'|'+region)
   if not spec['eligible']:records.append(r|dict(status='ineligible',reason=pair['source_ineligibility_reason']));continue
   c=caps[ep];other=caps[1-ep];positions=spec['recipient_positions'];donorpos=spec['donor_positions'];delta=torch.zeros_like(c['writes'][layer]);delta[:,positions]=other['writes'][layer][:,donorpos]-c['writes'][layer][:,positions];noop=not bool(delta.count_nonzero());base=c['logits'];blp=base.double().log_softmax(-1);g=pair['answer_groups'][ep];v=pair['variants'][ep];bm=margin(base,g,v)
   if noop:out=base
   else:logits,_=trace(model,c['ids'],replace=lambda j,z,w:w+delta if j==layer else w);calls+=1;out=logits[0,-1]
   olp=out.double().log_softmax(-1);effect=margin(out,g,v)-bm;energy=float((blp.exp()*(olp-blp).square()).sum());records.append(r|dict(status='evaluated',noop=noop,delta_energy=float(delta.double().square().sum()),native_write_energy=float(c['writes'][layer].double().square().sum()),response_energy=energy,output_KL=float((blp.exp()*(blp-olp)).sum()),native_margin=bm,margin_change=effect,native_binary=bm>0,native_top1=int(base.argmax()) in g[v],changed_top1=int(out.argmax()) in g[v],positions_count=len(positions)));samples['changed'][r['sample_key']]=out.cpu();samples['deltas'][r['sample_key']]=dict(recipient=c['writes'][layer][:,positions].cpu(),donor=other['writes'][layer][:,donorpos].cpu(),delta=delta[:,positions].cpu())
  if (pi+1)%10==0:print('PAIRS',pi+1,'FORWARDS',calls,flush=True)
 summaries=[]
 for family in sorted({p['family'] for p in q['pairs']}):
  for layer in q['layers']:
   for region in ['changed_input_positions','final_position','identity']:
    allrows=[r for r in records if (r['family'],r['layer'],r['region'])==(family,layer,region)];rs=[r for r in allrows if r['status']=='evaluated'];n=len(rs);summaries.append(dict(family=family,layer=layer,region=region,assigned=len(allrows),evaluated=n,ineligible=len(allrows)-n,noops=sum(r['noop'] for r in rs),mean_KL=sum(r['output_KL'] for r in rs)/n if n else None,response_energy=sum(r['response_energy'] for r in rs),delta_energy=sum(r['delta_energy'] for r in rs),mean_absolute_margin_change=sum(abs(r['margin_change']) for r in rs)/n if n else None,margin_reductions=sum(r['margin_change']<0 for r in rs),native_binary_correct=sum(r['native_binary'] for r in rs)))
 assert len(records)==84;sp=A/'emlp_m59_native_samples.pt';assert not sp.exists();torch.save(samples,sp);op.write_text(json.dumps(dict(status='completed',protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,records=records,summaries=summaries,samples_path=str(sp),samples_sha256=sha(sp)),indent=2)+'\n');print('DONE',calls,len(records),flush=True)
if __name__=='__main__':main()

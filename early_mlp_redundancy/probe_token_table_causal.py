"""M5: opposite-document-half native product-write table interventions."""
import json,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from inspect_model import load_model,trace
P=Path(__file__).resolve().parent;A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def score(ref,out,ids):
 lp=ref.double().log_softmax(-1);lq=out.double().log_softmax(-1)
 return dict(KL=float((lp.exp()*(lp-lq)).sum(-1).mean()),top1_agreement=float((ref.argmax(-1)==out.argmax(-1)).double().mean()),CE=float(F.cross_entropy(out[:,:-1].double().reshape(-1,out.shape[-1]),ids[:,1:].reshape(-1))))
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 cp=P/'validation/natural_token_context_cohort.json';rp=P/'validation/natural_token_context.json';c=json.loads(cp.read_text());m4=json.loads(rp.read_text());cache=Path(m4['cache']);assert sha(cache)==m4['cache_sha256'];assert sha(cp)==m4['cohort_sha256']
 protocol=dict(cohort_sha256=sha(cp),M4_sha256=sha(rp),source_sha256=sha(Path(__file__)),design='Fixed64 reused training windows; opposite32-document-half current-token means fitted on M4 deduplicated native product writes. Intervene one layer0/1/2 at a time, every covered position, retain Down bias and exact native writes at uncovered positions. Norm control scales LIVE native product write to table product norm. All50304logits KL; next-token CE on127positions. Native baseline64 plus384interventions. No refit, heldout task or toy sweep. Save full outputs and local writes for documents0/32 for independent audit.',limitations='Table errors mix context dependence and finite-sample estimation error. Norm-only control requires native computation. Direction/magnitude effects are not additive. Output metrics include uncovered positions which can receive downstream effects. Not native channel pruning.')
 (P/'validation/token_table_causal_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 saved=torch.load(cache,weights_only=True);tok=saved['tokens'];docs=saved['documents'];uni=saved['unique_indices'];tables={};covered={}
 for half in [0,1]:
  tr=uni[docs[uni]//32!=half];types,inv,counts=torch.unique(tok[tr],return_inverse=True,return_counts=True);covered[half]=types
  for l in range(3):
   y=saved['Y'][l].double();mu=torch.zeros(len(types),1152,dtype=torch.float64).index_add_(0,inv,y[tr])/counts[:,None];table=torch.zeros(50257,1152);table[types]=mu.float();tables[half,l]=table
 model,receipt=load_model();device=next(model.parameters()).device;assert receipt.weights_sha256==m4['checkpoint']['weights_sha256'];records=[];samples=[];replay=[];calls=0
 for d,ids0 in enumerate(c['token_ids']):
  ids=torch.tensor([ids0],device=device);native,cap=trace(model,ids);calls+=1;base=score(native,native,ids);half=d//32;mask=torch.isin(ids,covered[half].to(device));sample=dict(document=d,ids=ids.cpu(),native=native.cpu(),conditions=[]) if d in [0,32] else None
  for l in range(3):
   mlp=model.transformer.h[l].mlp;product=mlp.Down(cap[l]['products']);old=saved['Y'][l][d*128:(d+1)*128].to(device);err=float((product[0]-old).abs().max());replay.append(err);assert torch.equal(product[0],old)
   pred=tables[half,l][torch.tensor(ids0)].to(device)[None];pn=product.norm(dim=-1,keepdim=True);assert float(pn.min())>1e-12
   for kind in ['table','norm_only']:
    local={}
    def replace(i,z,write):
     if i!=l:return write
     # Recompute the product explicitly to avoid subtracting the bias numerically.
     v=mlp.Down(mlp.Left(z)*mlp.Right(z));proposed=pred if kind=='table' else v*(pred.norm(dim=-1,keepdim=True)/v.norm(dim=-1,keepdim=True).clamp_min(1e-12));effective=torch.where(mask[:,:,None],proposed+mlp.Down_bias,write)
     assert torch.equal(effective[~mask],write[~mask]);local['effective']=effective.detach();return effective
    out,cc=trace(model,ids,replace=replace);calls+=1;met=score(native,out,ids);effective=local['effective'];target_delta=(effective-cap[l]['native_write']).double();local_error=float(target_delta.square().sum()/product.double().square().sum());rec=dict(document=d,layer=l,kind=kind,covered_positions=int(mask.sum()),positions=128,local_relative_squared_error=local_error,native_CE=base['CE'],delta_CE=met['CE']-base['CE'],**met);records.append(rec)
    if sample is not None:sample['conditions'].append(dict(layer=l,kind=kind,output=out.cpu(),native_product=product.cpu(),native_write=cap[l]['native_write'].cpu(),table_product=pred.cpu(),effective_write=effective.cpu(),bias=mlp.Down_bias.cpu(),mask=mask.cpu()))
  if sample is not None:samples.append(sample)
  if d%8==7:print('DOCUMENTS',d+1,flush=True)
 summaries=[]
 for l in range(3):
  for kind in ['table','norm_only']:
   rs=[x for x in records if x['layer']==l and x['kind']==kind];summaries.append(dict(layer=l,kind=kind,**{key:sum(x[key] for x in rs)/len(rs) for key in ['KL','top1_agreement','CE','native_CE','delta_CE','local_relative_squared_error','covered_positions']}))
 sp=A/'emlp_m5_causal_samples.pt';torch.save(samples,sp);out=dict(status='completed',protocol_sha256=sha(P/'validation/token_table_causal_protocol.json'),forward_count=calls,M4_replay_max_absolute_error=max(replay),audit_samples=str(sp),audit_samples_sha256=sha(sp),records=records,summaries=summaries);(P/'validation/token_table_causal.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()

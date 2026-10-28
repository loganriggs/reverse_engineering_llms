"""Fixed random-partition native MLP1 deletion diagnostic, without ranking or refitting.
Four disjoint quarters, two complementary halves, all products, and entire write.
Opened 80-pair training bank, all160 endpoints retained. Not corpus generalization.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
from inspect_model import load_model,trace
from probe_training_task_responses import competence,kl
P=Path(__file__).resolve().parent

def metrics(native,edited,ids):
 p=native.double().log_softmax(-1);q=edited.double().log_softmax(-1)
 k=(p.exp()*(p-q)).sum(-1)
 target=ids[:,1:,None];ce=(-q[:,:-1].gather(-1,target)+p[:,:-1].gather(-1,target)).squeeze(-1)
 return dict(mean_KL=float(k.mean()),max_position_KL=float(k.max()),last_KL=float(k[0,-1]),observed_next_token_dCE=float(ce.mean()),top1_agreement=float((native.argmax(-1)==edited.argmax(-1)).float().mean()))

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 inp=P/'validation/training_source_context_audit.json';panel=json.loads(inp.read_text());assert len(panel['rows'])==80
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');model,receipt=load_model();device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp
 n=mlp.Down.weight.shape[1];assert n==4608
 perm=torch.randperm(n,generator=torch.Generator().manual_seed(2026102901));groups=list(perm.chunk(4))
 sets={f'quarter_{i}':g.tolist() for i,g in enumerate(groups)}
 sets.update(half_01=torch.cat(groups[:2]).tolist(),half_23=torch.cat(groups[2:]).tolist(),all_products=list(range(n)),whole_write=list(range(n)))
 rows=[];samples=[];nativechecks=[];calls=0;identity_error=0.
 for pi,pair in enumerate(panel['rows']):
  for ep in (0,1):
   ids=torch.tensor([pair['token_ids'][ep]],device=device);base,c=trace(model,ids);calls+=1
   nativechecks.append(dict(id=pair['id'],family=pair['family'],endpoint=ep,**competence(base[0,-1],pair['answer_groups'][ep],int(pair['variants'][ep]))))
   for kind,indices in sets.items():
    keep=torch.ones(n,device=device);keep[indices]=0
    def patch(i,z,y):
     if i!=1:return y
     if kind=='whole_write':return torch.zeros_like(y)
     products=mlp.Left(z)*mlp.Right(z)
     return mlp.Down(products*keep)+mlp.Down_bias
    lg,ec=trace(model,ids,replace=patch);calls+=1
    if kind=='all_products':assert torch.equal(ec[1]['effective_write'],mlp.Down_bias.expand_as(ec[1]['effective_write']))
    if kind=='whole_write':assert torch.count_nonzero(ec[1]['effective_write'])==0
    row=dict(id=pair['id'],family=pair['family'],endpoint=ep,kind=kind,removed_channels=len(indices),**metrics(base,lg,ids),task=competence(lg[0,-1],pair['answer_groups'][ep],int(pair['variants'][ep])))
    rows.append(row)
    if pi==0 and ep==0:samples.append(dict(row=len(rows)-1,native=base.cpu(),edited=lg.cpu(),ids=ids.cpu()))
   if pi==0 and ep==0:
    def identity(i,z,y):return mlp.Down(mlp.Left(z)*mlp.Right(z))+mlp.Down_bias if i==1 else y
    check,_=trace(model,ids,replace=identity);calls+=1;identity_error=float((base-check).abs().max());assert identity_error==0
  print('completed',pi+1,80,flush=True)
 summaries=[]
 for family in [None]+list(panel['families']):
  for kind in sets:
   rr=[r for r in rows if r['kind']==kind and (family is None or r['family']==family)]
   summaries.append(dict(family=family,kind=kind,n=len(rr),**{f:sum(r[f] for r in rr)/len(rr) for f in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement']},worst_endpoint_mean_KL=max(r['mean_KL'] for r in rr),binary_rate=sum(r['task']['binary'] for r in rr)/len(rr),task_top1_rate=sum(r['task']['top1'] for r in rr)/len(rr)))
 out=dict(status='completed_diagnostic',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),panel_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),checkpoint_sha256=receipt.weights_sha256,seed=2026102901,removed_sets=sets,forward_count=calls,identity_max_logit_error=identity_error,native=nativechecks,summaries=summaries,rows=rows,
 protocol='Zero selected native products at all positions, preserve Down bias except whole_write. No refit/rescaling/selection. Fixed one random four-way partition. All160 opened training endpoints, including errors. Equal endpoint-weighted summaries; observed-token CE describes prompt continuations, not a natural-corpus benchmark. KL covers all50304 logits and all positions. No preservation threshold or population claim.')
 args.output.write_text(json.dumps(out,indent=2)+'\n');torch.save(samples,'/tmp/emlp_d1_deletion_logits.pt');print(json.dumps([s for s in summaries if s['family'] is None],indent=2))
if __name__=='__main__':main()

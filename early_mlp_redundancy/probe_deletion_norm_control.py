"""D2: preserve native MLP1 direction while matching fixed deleted-write norms."""
import argparse,hashlib,json
from pathlib import Path
import torch
from inspect_model import load_model,trace
from probe_mlp1_group_deletion import metrics
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 inp=P/'validation/training_source_context_audit.json';panel=json.loads(inp.read_text());dpath=P/'validation/mlp1_group_deletion.json';d1=json.loads(dpath.read_text())
 sets={k:v for k,v in d1['removed_sets'].items() if k.startswith(('quarter','half'))};assert len(sets)==6
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');m,receipt=load_model();device=next(m.parameters()).device;mlp=m.transformer.h[1].mlp
 rows=[];samples=[];calls=0;replay=[];maxnormerror=0.;zeros=0
 prior={(r['id'],r['endpoint'],r['kind']):r for r in d1['rows']}
 for pi,pair in enumerate(panel['rows']):
  for ep in (0,1):
   ids=torch.tensor([pair['token_ids'][ep]],device=device);base,c=trace(m,ids);calls+=1;y=c[1]['native_write'];prod=c[1]['products'];yn=y.norm(dim=-1,keepdim=True)
   for kind,indices in sets.items():
    keep=torch.ones(4608,device=device);keep[indices]=0
    deleted=mlp.Down(prod*keep)+mlp.Down_bias;dn=deleted.norm(dim=-1,keepdim=True)
    zero=yn==0;zeros+=int(zero.sum());assert not bool((zero & (dn>0)).any()),'Nonzero deleted write has no native direction; control undefined.'
    ratio=dn/yn.clamp_min(torch.finfo(y.dtype).tiny);control=torch.where(zero,torch.zeros_like(y),y*ratio)
    normerror=float(((control.norm(dim=-1,keepdim=True)-dn).abs()/dn.clamp_min(1e-12)).max());maxnormerror=max(maxnormerror,normerror);assert normerror<1e-6
    def patch(i,z,v):return control if i==1 else v
    lg,_=trace(m,ids,replace=patch);calls+=1
    cosine=torch.nn.functional.cosine_similarity(y,deleted,dim=-1)
    row=dict(id=pair['id'],family=pair['family'],endpoint=ep,kind=kind,**metrics(base,lg,ids),task=competence(lg[0,-1],pair['answer_groups'][ep],int(pair['variants'][ep])),mean_deleted_native_norm_ratio=float(ratio.mean()),mean_deleted_native_cosine=float(cosine.mean()),relative_deleted_write_squared_error=float((deleted-y).square().sum()/y.square().sum()),relative_control_write_squared_error=float((control-y).square().sum()/y.square().sum()))
    rows.append(row)
    if pi==0 and ep==0:samples.append(dict(row=len(rows)-1,native=base.cpu(),edited=lg.cpu(),ids=ids.cpu(),native_write=y.cpu(),deleted_write=deleted.cpu(),control_write=control.cpu()))
    if kind=='quarter_0':
     def replaypatch(i,z,v):return deleted if i==1 else v
     check,_=trace(m,ids,replace=replaypatch);calls+=1;met=metrics(base,check,ids);old=prior[(pair['id'],ep,kind)]
     replay.extend(abs(met[f]-old[f]) for f in met)
  print('completed',pi+1,80,flush=True)
 summaries=[]
 for family in [None]+list(panel['families']):
  for kind in sets:
   rr=[r for r in rows if r['kind']==kind and (family is None or r['family']==family)]
   fields=['mean_KL','last_KL','observed_next_token_dCE','top1_agreement','mean_deleted_native_norm_ratio','mean_deleted_native_cosine','relative_deleted_write_squared_error','relative_control_write_squared_error']
   old=[prior[(r['id'],r['endpoint'],r['kind'])] for r in rr]
   summaries.append(dict(family=family,kind=kind,n=len(rr),**{f:sum(r[f] for r in rr)/len(rr) for f in fields},deletion_mean_KL=sum(r['mean_KL'] for r in old)/len(old),binary_rate=sum(r['task']['binary'] for r in rr)/len(rr),task_top1_rate=sum(r['task']['top1'] for r in rr)/len(rr)))
 assert max(replay)==0
 out=dict(status='completed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),panel_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),D1_sha256=hashlib.sha256(dpath.read_bytes()).hexdigest(),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,zero_native_write_positions=zeros,max_norm_relative_error=maxnormerror,quarter0_replay_metric_max_error=max(replay),summaries=summaries,rows=rows,
 protocol='Same six quarter/half masks and160 opened training endpoints as D1. Per-position control y*norm(y_deleted)/norm(y), bias included, native direction retained. Zero native and deleted norms yield zero; undefined nonzero deleted norm with zero native norm aborts. Controls use native computation and are not pruned models. All-position full-output metrics, equal endpoint weighting, no gain/mask tuning. A KL difference is not a unique additive causal attribution.')
 args.output.write_text(json.dumps(out,indent=2)+'\n');torch.save(samples,'/tmp/emlp_d2_samples.pt');print(json.dumps([r for r in summaries if r['family'] is None],indent=2))
if __name__=='__main__':main()

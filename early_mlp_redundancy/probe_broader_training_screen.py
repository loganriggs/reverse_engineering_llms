"""M55 fixed training-family screen; no representation fits or prompt replacement."""
from pathlib import Path
import json,hashlib
import torch
from inspect_model import load_model,trace
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);mp=V/'broader_training_screen_manifest.json';m=json.loads(mp.read_text());mh=sha(mp);design=json.loads((V/'broader_training_screen_design.json').read_text());assert m['design_sha256']==sha(V/'broader_training_screen_design.json');op=V/'broader_training_screen.json';assert not op.exists();model,receipt=load_model();rows=[];saved=[];replays=[];calls=0
 for p in m['pairs']:
  if not p['valid']:continue
  endpoints=[]
  for ep,e in enumerate(p['endpoints']):
   ids=torch.tensor([e['ids']],device='cuda');logits,_=trace(model,ids);calls+=1;last=logits[0,-1].double();groups=e['answer_groups'];margin=float(torch.logsumexp(last[groups[ep]],0)-torch.logsumexp(last[groups[1-ep]],0));top=int(last.argmax());endpoints.append(dict(endpoint=ep,margin=margin,binary=margin>0,top1=top in groups[ep],top_token=top));saved.append(dict(id=p['id'],endpoint=ep,logits=last.cpu()))
   if p['id'].endswith(':0') and ep==0:
    heads=[];handle=model.lm_head.register_forward_hook(lambda module,inputs,output:heads.append(output.detach()))
    try:model(ids,ids)
    finally:handle.remove()
    direct=30*torch.tanh(heads[0]/30);err=float((direct-logits).abs().max());assert err<1e-4;replays.append(dict(id=p['id'],max_absolute_error=err));calls+=1
  rows.append(dict(id=p['id'],family=p['family'],format=p['format'],equal_length=p['equal_length'],endpoints=endpoints,both_correct=all(e['binary'] for e in endpoints),top1_fraction=sum(e['top1'] for e in endpoints)/2))
 summaries=[]
 for family in design['candidate_families']:
  for fmt in design['formats']:
   rr=[r for r in rows if r['family']==family and r['format']==fmt];n=len(rr);acc=sum(r['both_correct'] for r in rr)/n if n else 0;top=sum(r['top1_fraction'] for r in rr)/n if n else 0;t=design['threshold'];passed=n>=t['min_valid_distinct_pairs'] and acc>=t['min_both_endpoint_accuracy'] and top>=t['min_intended_vocab_top1'];summaries.append(dict(family=family,format=fmt,valid_pairs=n,equal_length_pairs=sum(r['equal_length'] for r in rr),both_endpoint_accuracy=acc,top1=top,pass_gate=passed))
 assert sha(mp)==mh;sp=A/'emlp_m55_competence_logits.pt';assert not sp.exists();torch.save(saved,sp);op.write_text(json.dumps(dict(status='completed',manifest_sha256=mh,source_sha256=sha(Path(__file__)),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,rows=rows,summaries=summaries,native_replays=replays,logits_path=str(sp),logits_sha256=sha(sp),scope='Training-only competence, no causal/representation claim. Unmatched token lengths reported and not excluded by this competence gate; alignment remains a later causal-design constraint.'),indent=2)+'\n');print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()

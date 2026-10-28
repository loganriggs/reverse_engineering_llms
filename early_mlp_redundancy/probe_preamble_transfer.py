"""Compare context-specific and original-context MLP1 responses on competent preamble cohorts."""
import argparse, hashlib, json
from pathlib import Path
import torch
from inspect_model import load_model, trace
from probe_training_task_responses import kl, margin
P=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 inp=P/'validation/preamble_competence.json'; screen=json.loads(inp.read_text())
 panel=P/'validation/training_source_context_audit.json'; original={p['id']:p for p in json.loads(panel.read_text())['rows']}
 passed={(c['family'],c['preamble_index']) for c in screen['cohorts'] if c['competence_gate_pass'] and c['all_single_token']}
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');model,receipt=load_model();device=next(model.parameters()).device
 rows=[];vectors=[];samples=[];calls=0
 def states(tokens):
  nonlocal calls
  out=[]
  for t in tokens:
   ids=torch.tensor([t],device=device);lg,c=trace(model,ids);calls+=1;out.append((ids,lg,c))
  return out
 def response(st,rec,pos):
  nonlocal calls
  def patch(i,z,y):
   if i!=0:return y
   out=y.clone();out[:,pos]=st[1-rec][2][0]['native_write'][:,pos];return out
  _,c=trace(model,st[rec][0],replace=patch);calls+=1
  return c[1]['native_write'][:,pos]-st[rec][2][1]['native_write'][:,pos]
 for pi,pair in enumerate(screen['rows']):
  if (pair['family'],pair['preamble_index']) not in passed:continue
  orig=original[pair['id']];old=states(orig['token_ids']);new=states(pair['token_ids']);pos=pair['changed_positions'][0]
  for rec in (0,1):
   a=response(old,rec,orig['source_position']);b=response(new,rec,pos)
   assert a.norm()>0 and b.norm()>0
   vectors.append(dict(id=pair['id'],preamble_index=pair['preamble_index'],receiver=rec,old=a.cpu(),new=b.cpu()))
   base=new[rec][1][0,-1];teacher=new[1-rec][1][0,-1];groups=orig['answer_groups'][1-rec];variant=int(orig['variants'][1-rec])
   edits=[('context_specific',b),('original_context',a),('original_norm_matched',a*(b.norm()/a.norm()))]
   for k in range(4):
    g=torch.Generator(device=device).manual_seed(2026110200+100*pi+10*rec+k)
    perm=torch.randperm(b.shape[-1],device=device,generator=g);sign=torch.randint(0,2,b.shape,device=device,generator=g)*2-1
    edits.append(('random',b[:,perm]*sign))
   for ei,(kind,edit) in enumerate(edits):
    def patch(i,z,y):
     if i!=1:return y
     out=y.clone();out[:,pos]+=edit;return out
    lg,_=trace(model,new[rec][0],replace=patch);calls+=1;candidate=lg[0,-1]
    rows.append(dict(id=pair['id'],family=pair['family'],preamble_index=pair['preamble_index'],receiver=rec,kind=kind,
      response_cosine=float(torch.nn.functional.cosine_similarity(a,b).item()),old_norm=float(a.norm()),new_norm=float(b.norm()),
      relative_vector_error=float((a-b).square().sum()/b.square().sum()),native_donor_KL=kl(teacher,base),remaining_donor_KL=kl(teacher,candidate),
      donor_margin_gap=margin(teacher,groups,variant)-margin(base,groups,variant),margin_gain=margin(candidate,groups,variant)-margin(base,groups,variant)))
    if len(vectors)==1:samples.append(dict(row=len(rows)-1,teacher=teacher.cpu(),baseline=base.cpu(),candidate=candidate.cpu(),groups=groups,variant=variant))
  print('completed',pair['preamble_index'],pair['id'],flush=True)
 cohorts=[]
 for family,pre in sorted(passed):
  kinds={}
  for kind in ('context_specific','original_context','original_norm_matched','random'):
   rr=[r for r in rows if r['family']==family and r['preamble_index']==pre and r['kind']==kind]
   kinds[kind]=dict(n=len(rr),remaining_donor_KL_ratio=sum(r['remaining_donor_KL'] for r in rr)/sum(r['native_donor_KL'] for r in rr),margin_recovery=sum(r['margin_gain'] for r in rr)/sum(r['donor_margin_gap'] for r in rr),mean_cosine=sum(r['response_cosine'] for r in rr)/len(rr))
  cohorts.append(dict(family=family,preamble_index=pre,responses=kinds))
 result=dict(status='completed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),competence_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,cohorts=cohorts,rows=rows,
 protocol='Only complete passing family/preamble cohorts, including their incorrect examples. Generate local MLP1 response under donor MLP0 write separately in original and prefixed context; apply each to prefixed recipient. Also norm-match original response and use four signed-permutation controls. No fitted parameters; original response is an oracle for the same training token pair, not a token-only learned model.')
 args.output.write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(vectors=vectors,samples=samples),'/tmp/emlp_p8_transfer_samples.pt');print(json.dumps(cohorts,indent=2))
if __name__=='__main__':main()

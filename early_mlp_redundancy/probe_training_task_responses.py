"""EMLP-P7: training-only competence and MLP1 source-response task-coverage screen."""
import argparse,json,hashlib
from pathlib import Path
import torch
from inspect_model import load_model,trace
P=Path(__file__).resolve().parent

def kl(a,b):
 a=a.double().log_softmax(-1);b=b.double().log_softmax(-1);return float((a.exp()*(a-b)).sum())
def margin(lg,groups,variant):
 return float(torch.logsumexp(lg[groups[variant]].double(),0)-torch.logsumexp(lg[groups[1-variant]].double(),0))
def competence(lg,groups,variant):
 return dict(binary=margin(lg,groups,variant)>0,top1=int(lg.argmax()) in groups[variant],margin=margin(lg,groups,variant),top_token=int(lg.argmax()))

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');m,receipt=load_model();device=next(m.parameters()).device
 inp=P/'validation/training_source_context_audit.json';panel=json.loads(inp.read_text());rows=[];native=[];calls=0;identity=0.;saved=[]
 for pi,pair in enumerate(panel['rows']):
  states=[]
  for ids in pair['token_ids']:
   ids=torch.tensor([ids],device=device,dtype=torch.long);lg,c=trace(m,ids);calls+=1;states.append((ids,lg,c))
  checks=[competence(states[i][1][0,-1],pair['answer_groups'][i],int(pair['variants'][i])) for i in (0,1)]
  native.append(dict(id=pair['id'],family=pair['family'],single_token_edit=pair['single_token_edit'],endpoints=checks,both_binary=all(c['binary'] for c in checks),top1_rate=sum(c['top1'] for c in checks)/2))
  if not pair['single_token_edit']:continue
  pos=pair['source_position'];assert len(pair['changed_positions'])==1
  for rec in (0,1):
   ids,base,c=states[rec];_,donor,dc=states[1-rec];groups=pair['answer_groups'][1-rec];target_variant=int(pair['variants'][1-rec])
   # Only local MLP0 change is needed to generate the MLP1 response at this position.
   # By causality, later MLP0 changes cannot affect this write.
   def source(i,z,y):
    if i!=0:return y
    out=y.clone();out[:,pos]=dc[0]['native_write'][:,pos];return out
   _,ic=trace(m,ids,replace=source);calls+=1
   response=ic[1]['native_write'][:,pos]-c[1]['native_write'][:,pos]
   baseline=base[0,-1];teacher=donor[0,-1];gap=kl(teacher,baseline);mgap=margin(teacher,groups,target_variant)-margin(baseline,groups,target_variant)
   for control in range(5):
    if control==0:edit=response;kind='source_response'
    else:
     g=torch.Generator(device=device).manual_seed(2026110100+pi*100+rec*10+control)
     perm=torch.randperm(response.shape[-1],device=device,generator=g);sign=torch.randint(0,2,response.shape,device=device,generator=g)*2-1
     edit=response[:,perm]*sign;kind='random_source_response'
     assert torch.allclose(edit.norm(),response.norm(),rtol=1e-6,atol=1e-6)
    def patch(i,z,y):
     if i!=1:return y
     out=y.clone();out[:,pos]+=edit;return out
    lg,_=trace(m,ids,replace=patch);calls+=1;candidate=lg[0,-1]
    rows.append(dict(id=pair['id'],family=pair['family'],receiver=rec,kind=kind,control=control,source_position=pos,
      source_response_norm=float(response.norm()),native_donor_KL=gap,remaining_donor_KL=kl(teacher,candidate),recipient_change_KL=kl(baseline,candidate),
      donor_margin_gap=mgap,margin_gain=margin(candidate,groups,target_variant)-margin(baseline,groups,target_variant)))
    if pi==0 and rec==0 and control in (0,1):saved.append(dict(row=len(rows)-1,teacher=teacher.cpu(),baseline=baseline.cpu(),candidate=candidate.cpu(),groups=groups,variant=target_variant))
  print('completed',pi+1,len(panel['rows']),pair['family'],flush=True)
 families={}
 for family in panel['families']:
  cohort=[r for r in native if r['family']==family];eligible=[r for r in cohort if r['single_token_edit']]
  binary=sum(r['both_binary'] for r in cohort)/len(cohort);top=sum(r['top1_rate'] for r in cohort)/len(cohort)
  fm=dict(native_pairs=len(cohort),both_endpoint_binary_rate=binary,full_vocab_top1_rate=top,
    competence_gate_pass=binary>=.75 and top>=.5,aligned_pairs=len(eligible),source_diversity=panel['families'][family],responses={})
  for kind in ('source_response','random_source_response'):
   rr=[r for r in rows if r['family']==family and r['kind']==kind]
   if not rr:continue
   kg=sum(r['native_donor_KL'] for r in rr);mg=sum(r['donor_margin_gap'] for r in rr)
   fm['responses'][kind]=dict(n=len(rr),remaining_donor_KL_ratio=sum(r['remaining_donor_KL'] for r in rr)/kg,
      donor_margin_recovery=sum(r['margin_gain'] for r in rr)/mg if abs(mg)>1e-12 else None,
      mean_recipient_change_KL=sum(r['recipient_change_KL'] for r in rr)/len(rr),mean_native_donor_KL=kg/len(rr),mean_native_margin_gap=mg/len(rr))
  families[family]=fm
 result=dict(status='completed_training_only_screen',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),panel_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),checkpoint_sha256=receipt.weights_sha256,
  protocol='Exact80previously-opened training pairs. Native competence rechecked on160endpoints. Apply source-position MLP1 response induced by donor MLP0write only for76single-token pairs; four same-site normmatched random controls. No per-example correctness filtering, no fits, no evaluation rows.',
  forward_count=calls,families=families,native=native,rows=rows,
  limitations=['Training diagnostic, not held-out transfer.','Most families lack multiple pre-edit contexts for a fixed token pair.','Cohort gate >=75%both-endpoint binary and >=50%full-vocab top1; wrong examples retained.','number_successor has multi-token changes and is competence-only, not a failed response test.','Native donor response is an oracle target, not a learned generator.'])
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');torch.save(saved,'/tmp/emlp_p7_logits.pt')
 print(json.dumps({f:{'competent':v['competence_gate_pass'],'responses':v['responses']} for f,v in families.items()},indent=2))
if __name__=='__main__':main()

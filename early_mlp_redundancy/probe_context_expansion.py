"""M17 native competence on all frozen expanded contexts; no training."""
from pathlib import Path
import json,hashlib
import torch
from inspect_model import load_model,trace
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');pp=P/'context_expansion_protocol.json';p=json.loads(pp.read_text());op=P/'context_expansion_competence.json';assert not op.exists();model,receipt=load_model();device=next(model.parameters()).device;results=[];samples=[]
 for i,row in enumerate(p['rows']):
  endpoints=[]
  for ep in [0,1]:
   ids=torch.tensor([row['token_ids'][ep]],device=device);lg,_=trace(model,ids);endpoints.append(competence(lg[0,-1],row['answer_groups'][ep],row['variants'][ep]));samples.append(dict(id=row['id'],endpoint=ep,logits=lg[0,-1].cpu()))
  results.append(dict(id=row['id'],family=row['family'],template=row['template'],template_split=row['template_split'],lexical_split=row['lexical_split'],endpoints=endpoints))
  if (i+1)%80==0:print('PAIRS',i+1,len(p['rows']),flush=True)
 cohorts=[]
 for template in p['templates']:
  for split in ['train','evaluation']:
   for family in sorted({r['family'] for r in results if r['lexical_split']==split}):
    rr=[r for r in results if r['template']==template['id'] and r['lexical_split']==split and r['family']==family];binary=sum(all(e['binary'] for e in r['endpoints']) for r in rr)/len(rr);top=sum(e['top1'] for r in rr for e in r['endpoints'])/(2*len(rr));cohorts.append(dict(template=template['id'],template_split=template['split'],lexical_split=split,family=family,pairs=len(rr),both_binary_rate=binary,top1_rate=top,pass_gate=binary>=.75 and top>=.5))
 ap=A/'emlp_m17_native_context_samples.pt';torch.save(samples,ap);out=dict(status='completed',protocol_sha256=sha(pp),probe_sha256=sha(Path(__file__)),checkpoint_sha256=receipt.weights_sha256,forward_count=len(samples),results=results,cohorts=cohorts,samples_path=str(ap),samples_sha256=sha(ap));op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',len(samples),'PASS',sum(c['pass_gate'] for c in cohorts),'OF',len(cohorts),flush=True)
if __name__=='__main__':main()

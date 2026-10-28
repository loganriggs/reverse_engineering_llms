"""M9 frozen M8 featurizers on the existing ten-family development task bank."""
import json,hashlib
from pathlib import Path
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from probe_mlp1_group_deletion import metrics
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent;A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');mp=P/'validation/causal_featurizers.json';rp=P/'validation/training_source_context_audit.json';m8=json.loads(mp.read_text());panel=json.loads(rp.read_text());assert len(panel['rows'])==80;excluded={'sequence_copy','spatial_inverse','weekday_successor','role_binding','tense'};assert not excluded&set(panel['families'])
 protocol=dict(source_sha256=sha(Path(__file__)),M8_sha256=sha(mp),panel_sha256=sha(rp),models=m8['models'],excluded_families=sorted(excluded),design='Eight frozen scalar/block MSE/KL models, seeds71/72. Reconstruct live native MLP1 product write at every position; retain bias. No fitting/checkpoint selection. All80pairs160endpoints, all10families incl native failures. Full50304logit KL andnativeTop1, prompt next-tokenCE change, answer-group logsumexp margin/binary/top1, pair correctness andmargin error, localwriteerror. Equalendpoint means plus equal-family macro; no masking to positive results.',scope='Natural-text-trained representations transferred to previously opened authored task development panel. Tasks absent from M7/M8fitting but historically examined. NOT pristine whole-task holdout or new-task discovery. All familyresults retained; original exclusions remain unused.')
 pp=P/'validation/featurizer_task_transfer_protocol.json';pp.write_text(json.dumps(protocol,indent=2)+'\n');model,receipt=load_model();device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;models=[]
 for entry in m8['models']:
  path=Path(entry['path']);assert sha(path)==entry['sha256'];state=torch.load(path,weights_only=True);f=Featurizer(entry['kind'],entry['seed']).to(device);f.load_state_dict(state['state']);f.eval();models.append((entry,f,state['mean'].to(device),state['scale'].to(device)))
 native=[];rows=[];samples=[];calls=0;sampled=set();old=json.loads((P/'validation/mlp1_group_deletion.json').read_text());replay=[]
 for pi,pair in enumerate(panel['rows']):
  take=pair['family'] not in sampled
  for ep in [0,1]:
   ids=torch.tensor([pair['token_ids'][ep]],device=device);base,cap=trace(model,ids);calls+=1;v=mlp.Down(cap[1]['products']);groups=pair['answer_groups'][ep];variant=int(pair['variants'][ep]);nt=competence(base[0,-1],groups,variant);native.append(dict(id=pair['id'],family=pair['family'],endpoint=ep,**nt));previous=next(x for x in old['native'] if x['id']==pair['id'] and x['endpoint']==ep);replay.append(abs(nt['margin']-previous['margin']));assert replay[-1]<1e-6
   for entry,f,mean,scale in models:
    reconstruction,support,code=f((v-mean)/scale);q=reconstruction*scale+mean;assert torch.all(support.sum(-1)==32)
    out,_=trace(model,ids,replace=lambda i,z,w:q+mlp.Down_bias if i==1 else w);calls+=1;task=competence(out[0,-1],groups,variant);local=float((q.double()-v.double()).square().sum()/v.double().square().sum());row=dict(id=pair['id'],family=pair['family'],endpoint=ep,kind=entry['kind'],seed=entry['seed'],objective=entry['objective'],**metrics(base,out,ids),local_relative_error=local,task=task,native_task=nt,absolute_margin_error=abs(task['margin']-nt['margin']),binary_decision_agreement=task['binary']==nt['binary']);rows.append(row)
    if take and ep==0:samples.append(dict(row=len(rows)-1,native=base.cpu(),output=out.cpu(),ids=ids.cpu(),groups=groups,variant=variant,native_product=v.cpu(),reconstructed_product=q.cpu()))
  sampled.add(pair['family'])
  if pi%10==9:print('PAIRS',pi+1,flush=True)
 summaries=[]
 for entry,_,_,_ in models:
  for family in [None]+list(panel['families']):
   rs=[x for x in rows if all(x[k]==entry[k] for k in ['kind','seed','objective']) and (family is None or x['family']==family)];pairs={x['id'] for x in rs};both=sum(all(x['task']['binary'] for x in rs if x['id']==pid) for pid in pairs);nativeboth=sum(all(x['native_task']['binary'] for x in rs if x['id']==pid) for pid in pairs)
   summaries.append({k:entry[k] for k in ['kind','seed','objective']}|dict(family=family,endpoints=len(rs),pairs=len(pairs),**{k:sum(x[k] for x in rs)/len(rs) for k in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement','local_relative_error','absolute_margin_error','binary_decision_agreement']},binary_rate=sum(x['task']['binary'] for x in rs)/len(rs),native_binary_rate=sum(x['native_task']['binary'] for x in rs)/len(rs),task_top1_rate=sum(x['task']['top1'] for x in rs)/len(rs),native_task_top1_rate=sum(x['native_task']['top1'] for x in rs)/len(rs),both_endpoint_binary_rate=both/len(pairs),native_both_endpoint_binary_rate=nativeboth/len(pairs)))
 macros=[]
 for entry,_,_,_ in models:
  rs=[x for x in summaries if x['family'] is not None and all(x[k]==entry[k] for k in ['kind','seed','objective'])];macros.append({k:entry[k] for k in ['kind','seed','objective']}|{k:sum(x[k] for x in rs)/len(rs) for k in ['mean_KL','last_KL','binary_rate','task_top1_rate','both_endpoint_binary_rate','absolute_margin_error']})
 sp=A/'emlp_m9_task_transfer_samples.pt';torch.save(samples,sp);out=dict(status='completed',protocol_sha256=sha(pp),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,native_margin_replay_max_error=max(replay),native=native,rows=rows,summaries=summaries,equal_family_macros=macros,samples_path=str(sp),samples_sha256=sha(sp));(P/'validation/featurizer_task_transfer.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([x for x in summaries if x['family'] is None],indent=2),flush=True)
if __name__=='__main__':main()

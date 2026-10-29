"""M72 all independent LS/original encoders on all32 natural evaluation documents, no fits."""
from pathlib import Path
import json,hashlib,torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from least_squares_encoder import LeastSquaresEncoder
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);pp=V/'balanced_response_evaluation_protocol.json';p=json.loads(pp.read_text());mp=V/'least_squares_evaluation_models.json';manifest=json.loads(mp.read_text());audit=json.loads((V/'least_squares_encoder_preflight.json').read_text());assert audit['status']=='passed' and audit['encoder_source_sha256']==sha(P/'least_squares_encoder.py');assert manifest['protocol_sha256']==sha(V/'least_squares_encoder_protocol.json');cp=V/'natural_token_context_cohort.json';assert sha(cp)==p['natural_cohort_sha256'];cohort=json.loads(cp.read_text());m4=json.loads((V/'natural_token_context.json').read_text());cache=Path(m4['cache']);assert sha(cache)==m4['cache_sha256'];writes=torch.load(cache,weights_only=True)['Y'][1];model,receipt=load_model();mlp=model.transformer.h[1].mlp;models=manifest['models'];states={}
 for e in models:
  path=Path(e['path']);assert sha(path)==e['sha256'];s=torch.load(path,weights_only=True);f=Featurizer(e['kind'],e['seed']).cuda();f.load_state_dict(s['state']);states[e['kind'],e['seed'],e['arm']]=LeastSquaresEncoder(f,s['mean'].cuda(),s['scale'].cuda())
 prior={(x['kind'],x['seed'],x['arm'],x['document']):x for x in json.loads((V/'encoded_eval_natural.json').read_text())['rows']};rows=[];files=[];calls=0;completed=0;progress=A/'emlp_m72_natural_progress.pt';dest=A/'emlp_m72_natural_samples';dest.mkdir(exist_ok=True);op=V/'least_squares_natural.json';assert not op.exists()
 if progress.exists():
  old=torch.load(progress,weights_only=True);assert old['protocol_sha256']==sha(pp) and old['models_sha256']==sha(mp) and old['source_sha256']==sha(Path(__file__));rows,files,calls,completed=[old[k] for k in ['rows','files','calls','completed']]
 for i,doc in enumerate(range(32,64)):
  if i<completed:continue
  ids=torch.tensor([cohort['token_ids'][doc]],device='cuda');base,cap=trace(model,ids);calls+=1;write=mlp.Down(cap[1]['products']);assert torch.equal(write[0].cpu(),writes[doc*128:(doc+1)*128]);nlp=base.double().log_softmax(-1);nativeCE=float(-nlp[:,:-1].gather(-1,ids[:,1:,None]).mean());sample=dict(document=doc,ids=ids.cpu(),native=base.cpu(),write=write.cpu(),conditions=[])
  for e in models:
   enc=states[e['kind'],e['seed'],e['arm']];q,support,code,info=enc.reconstruct(write,e['variant']);out,_=trace(model,ids,replace=lambda j,z,w:q+mlp.Down_bias if j==1 else w);calls+=1;lp=out.double().log_softmax(-1);CE=float(-lp[:,:-1].gather(-1,ids[:,1:,None]).mean());row={k:e[k] for k in ['kind','seed','arm']}|dict(document=doc,mean_KL=float((nlp.exp()*(nlp-lp)).sum(-1).mean()),native_CE=nativeCE,CE=CE,delta_CE=CE-nativeCE,top1_agreement=float((base.argmax(-1)==out.argmax(-1)).double().mean()),local_relative_error=float((write.double()-q.double()).square().sum()/write.double().square().sum()),active_mean=float(support.sum(-1).mean()));rows.append(row)
   if (e['kind'],e['seed'],e['arm'],doc) in prior:assert row==prior[e['kind'],e['seed'],e['arm'],doc]
   row['numerical_nonzero_mean']=info['numerical_nonzero_mean'];row['rank_mean']=info.get('rank_mean')
   if doc in [32,63]:sample['conditions'].append(dict(row=len(rows)-1,output=out.cpu(),reconstruction=q.cpu(),support=support.cpu(),code=code.cpu(),info=info))
  if doc in [32,63]:
   sp=dest/f'doc_{doc}.pt';torch.save(sample,sp);files.append(dict(path=str(sp),sha256=sha(sp),document=doc))
  tmp=progress.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(pp),models_sha256=sha(mp),source_sha256=sha(Path(__file__)),rows=rows,files=files,calls=calls,completed=i+1),tmp);tmp.replace(progress)
  if (i+1)%8==0:print('NATURAL_DOCS',i+1,'FORWARDS',calls,flush=True)
 summaries=[]
 for e in models:
  rr=[r for r in rows if all(r[k]==e[k] for k in ['kind','seed','arm'])];assert len(rr)==32;summaries.append({k:e[k] for k in ['kind','seed','arm']}|{k:sum(r[k] for r in rr)/32 for k in ['mean_KL','native_CE','CE','delta_CE','top1_agreement','local_relative_error','active_mean']})
 assert len(rows)==1728 and calls==1760;op.write_text(json.dumps(dict(status='completed',protocol_sha256=sha(pp),models_sha256=sha(mp),source_sha256=sha(Path(__file__)),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,rows=rows,summaries=summaries,files=files),indent=2)+'\n');print('DONE',calls,flush=True)
if __name__=='__main__':main()

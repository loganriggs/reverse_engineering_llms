"""M112 four fixed trained states on all32 reserved natural documents."""
import json
import torch
from fit_pca_encoder import V,A,sha
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from continuous_write_featurizer import ContinuousBlockFeaturizer


def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 rp=V/'continuous_training_results.json';r=json.loads(rp.read_text());audit=json.loads((V/'continuous_training_audit.json').read_text());assert audit['status']=='passed' and audit['results_sha256']==sha(rp)
 ep=V/'continuous_training_evaluation_protocol.json';assert r['receipt']['evaluation_protocol_sha256']==sha(ep)
 cohort=json.loads((V/'natural_token_context_cohort.json').read_text());nr=json.loads((V/'natural_token_context.json').read_text());assert sha(nr['cache'])==nr['cache_sha256'];oldwrites=torch.load(nr['cache'],weights_only=True)['Y'][1]
 states=[]
 for entry in r['models']:
  assert sha(entry['path'])==entry['sha256'];s=torch.load(entry['path'],weights_only=True);f=(Featurizer('block',entry['seed']) if entry['mode']=='hard' else ContinuousBlockFeaturizer(entry['seed'])).cuda();f.load_state_dict(s['state']);states.append((entry,f,s['mean'].cuda(),s['scale'].cuda()))
 previous={(x['seed'],x['document']):x for x in json.loads((V/'encoded_eval_natural.json').read_text())['rows'] if x['kind']=='block' and x['arm']=='KL_encoded_balanced_response'}
 model,mr=load_model();assert mr.weights_sha256==r['receipt']['checkpoint_sha256'];mlp=model.transformer.h[1].mlp;dest=A/'emlp_m112_natural';dest.mkdir(exist_ok=True);op=V/'continuous_natural.json';assert not op.exists();rows=[];files=[];replays=0
 for doc in range(32,64):
  ids=torch.tensor([cohort['token_ids'][doc]],device='cuda');base,cap=trace(model,ids);write=mlp.Down(cap[1]['products']);assert torch.equal(write[0].cpu(),oldwrites[doc*128:(doc+1)*128]);nlp=base.double().log_softmax(-1);nativece=float(-nlp[:,:-1].gather(-1,ids[:,1:,None]).mean());sample=dict(document=doc,ids=ids.cpu(),native=base.cpu(),write=write.cpu(),conditions=[])
  for entry,f,mu,scale in states:
   rec,support,code=f((write-mu)/scale);q=rec*scale+mu;out,_=trace(model,ids,replace=lambda i,z,w:q+mlp.Down_bias if i==1 else w);lp=out.double().log_softmax(-1);ce=float(-lp[:,:-1].gather(-1,ids[:,1:,None]).mean())
   row=dict(seed=entry['seed'],mode=entry['mode'],document=doc,mean_KL=float((nlp.exp()*(nlp-lp)).sum(-1).mean()),native_CE=nativece,CE=ce,delta_CE=ce-nativece,top1_agreement=float((base.argmax(-1)==out.argmax(-1)).double().mean()),local_relative_error=float((write.double()-q.double()).square().sum()/write.double().square().sum()),active_mean=float(support.sum(-1).mean()),code_energy=float(code.double().square().mean()))
   if entry['mode']=='hard':
    old=previous[entry['seed'],doc]
    for key in ['mean_KL','native_CE','CE','delta_CE','top1_agreement','local_relative_error','active_mean']:assert row[key]==old[key],key
    replays+=1
   rows.append(row);sample['conditions'].append(dict(row=row,output=out.cpu(),reconstruction=q.cpu(),code=code.cpu(),support=support.cpu()))
  path=dest/f'doc_{doc}.pt';assert not path.exists();torch.save(sample,path);files.append(dict(path=str(path),sha256=sha(path),document=doc));print('DOC',doc,flush=True)
 summaries=[]
 for entry in r['models']:
  rr=[x for x in rows if x['seed']==entry['seed'] and x['mode']==entry['mode']];assert len(rr)==32;summaries.append(dict(seed=entry['seed'],mode=entry['mode'])|{k:sum(x[k] for x in rr)/32 for k in ['mean_KL','native_CE','CE','delta_CE','top1_agreement','local_relative_error','active_mean','code_energy']})
 out=dict(status='completed_pending_audit',source_sha256=sha(__file__),models_sha256=sha(rp),evaluation_sha256=sha(ep),rows=rows,summaries=summaries,files=files,forwards=160,historical_hard_metric_replays=replays,native_write_replays=32)
 op.write_text(json.dumps(out,indent=2)+'\n');print('SUMMARIES',summaries,flush=True)
if __name__=='__main__':main()

"""M62 all frozen candidates on fixed source-group development holdouts."""
from pathlib import Path
import json,hashlib
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def margin(z,g,v):return float(torch.logsumexp(z.double()[g[v]],0)-torch.logsumexp(z.double()[g[1-v]],0))
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);pp=V/'balanced_eval_holdout_protocol.json';q=json.loads(pp.read_text());np=V/'source_holdout_native.json';npp=V/'source_holdout_native_protocol.json';mp=V/'balanced_response_training_results.json'
 for path,key in [(np,'native_results_sha256'),(npp,'native_protocol_sha256'),(mp,'models_sha256')]:assert sha(path)==q[key]
 native=json.loads(np.read_text());nprotocol=json.loads(npp.read_text());models=json.loads(mp.read_text())['models'];sp=Path(native['samples_path']);assert sha(sp)==native['samples_sha256'];saved=torch.load(sp,weights_only=True);op=V/'balanced_eval_holdout.json';assert not op.exists();dest=A/'emlp_m62_holdout_samples';dest.mkdir(exist_ok=True);model,receipt=load_model();assert receipt.weights_sha256==native['checkpoint_sha256'];mlp=model.transformer.h[1].mlp;states={}
 for e in models:
  path=Path(e['path']);assert sha(path)==e['sha256'];s=torch.load(path,weights_only=True);f=Featurizer(e['kind'],e['seed']).cuda();f.load_state_dict(s['state']);states[e['kind'],e['seed'],e['arm']]=(f,s['mean'].cuda(),s['scale'].cuda())
 rows=[];baselines=[];files=[];calls=0;completed=0;replays=0;progress=A/'emlp_m62_holdout_progress.pt';sourcehash=sha(Path(__file__))
 if progress.exists():
  old=torch.load(progress,weights_only=True);assert old['protocol_sha256']==sha(pp) and old['source_sha256']==sourcehash;rows,baselines,files,calls,completed,replays=[old[k] for k in ['rows','baselines','files','calls','completed','replays']]
 inputs=[(p,ep) for p in nprotocol['pairs'] for ep in [0,1]]
 for i,(pair,ep) in enumerate(inputs):
  if i<completed:continue
  key=pair['id']+f'|{ep}';ids=torch.tensor([pair['token_ids'][ep]],device='cuda');out,cap=trace(model,ids);calls+=1;base=out[0,-1];assert torch.equal(base.cpu(),saved['baselines'][key]['logits']);write=mlp.Down(cap[1]['products']);nlp=base.double().log_softmax(-1);g=pair['answer_groups'][ep];v=pair['variants'][ep];bm=margin(base,g,v);targets={};blob=dict(key=key,student_base={},student_changed={})
  nr=[r for r in native['records'] if r['pair_id']==pair['id'] and r['endpoint']==ep and r['layer']==1]
  for r in nr:
   if r['status']=='ineligible':continue
   delta=torch.zeros_like(write);delta[:,r['recipient_positions']]=saved['deltas'][r['sample_key']]['delta'].cuda();dh=hashlib.sha256(delta.cpu().numpy().tobytes()).hexdigest();no=not bool(delta.count_nonzero())
   if no:z=base
   else:out,_=trace(model,ids,replace=lambda j,x,w:w+delta if j==1 else w);calls+=1;z=out[0,-1]
   assert torch.equal(z.cpu(),saved['changed'][r['sample_key']]);replays+=1;resp=z.double().log_softmax(-1)-nlp;targets[r['region']]=(delta,dh,resp,r)
  for e in models:
   mk=e['kind'],e['seed'],e['arm'];f,mu,scale=states[mk];rec,support,_=f((write-mu)/scale);recon=rec*scale+mu;out,_=trace(model,ids,replace=lambda j,x,w:recon+mlp.Down_bias if j==1 else w);calls+=1;z=out[0,-1];lp=z.double().log_softmax(-1);bmargin=margin(z,g,v);ms=':'.join(map(str,mk));blob['student_base'][ms]=z.cpu();baselines.append({k:e[k] for k in ['kind','seed','arm']}|dict(key=key,family=pair['family'],final_KL=float((nlp.exp()*(nlp-lp)).sum()),active_mean=float(support.sum(-1).mean()),binary=bmargin>0,top1=int(z.argmax()) in g[v]))
   for r in nr:
    ident={k:e[k] for k in ['kind','seed','arm']}|dict(key=key,family=pair['family'],region=r['region'],status=r['status'],native_sample_key=r['sample_key'])
    if r['status']=='ineligible':rows.append(ident|dict(reason=r['reason']));continue
    delta,dh,resp,nrow=targets[r['region']]
    if nrow['noop']:changed=z
    else:out,_=trace(model,ids,replace=lambda j,x,w:recon+delta+mlp.Down_bias if j==1 else w);calls+=1;changed=out[0,-1]
    sr=changed.double().log_softmax(-1)-lp;effect=margin(changed,g,v)-bmargin;rows.append(ident|dict(delta_sha256=dh,response_error=float((nlp.exp()*(sr-resp).square()).sum()),target_energy=nrow['response_energy'],margin_error=(effect-nrow['margin_change'])**2,margin_target_energy=nrow['margin_change']**2));blob['student_changed'][ms+'|'+r['region']]=changed.cpu()
  path=dest/(hashlib.sha256(key.encode()).hexdigest()+'.pt');torch.save(blob,path);files.append(dict(key=key,path=str(path),sha256=sha(path)));tmp=progress.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(pp),source_sha256=sourcehash,rows=rows,baselines=baselines,files=files,calls=calls,completed=i+1,replays=replays),tmp);tmp.replace(progress)
  if (i+1)%10==0:print('INPUTS',i+1,'FORWARDS',calls,flush=True)
 summaries=[]
 for e in models:
  for region in ['changed_input_positions','final_position','identity']:
   for family in [None]+sorted({p['family'] for p in nprotocol['pairs']}):
    rr=[r for r in rows if all(r[k]==e[k] for k in ['kind','seed','arm']) and r['region']==region and (family is None or r['family']==family)];valid=[r for r in rr if r['status']=='evaluated'];en=sum(r['target_energy'] for r in valid);er=sum(r['response_error'] for r in valid);mn=sum(r['margin_target_energy'] for r in valid);me=sum(r['margin_error'] for r in valid);summaries.append({k:e[k] for k in ['kind','seed','arm']}|dict(region=region,family=family,assigned=len(rr),evaluated=len(valid),target_energy=en,error_energy=er,relative_response_error=er/en if en else None,margin_target_energy=mn,margin_error_energy=me,relative_margin_error=me/mn if mn else None))
 macros=[]
 for e in models:
  for region in ['changed_input_positions','final_position','identity']:
   rr=[r for r in summaries if all(r[k]==e[k] for k in ['kind','seed','arm']) and r['region']==region and r['family'] is not None];valid=[r for r in rr if r['relative_response_error'] is not None];macros.append({k:e[k] for k in ['kind','seed','arm']}|dict(region=region,families_defined=len(valid),families_undefined=[r['family'] for r in rr if r['relative_response_error'] is None],equal_family_response_error=sum(r['relative_response_error'] for r in valid)/len(valid) if valid else None))
 assert len(rows)==1512 and len(baselines)==504 and replays==84;op.write_text(json.dumps(dict(status='completed',protocol_sha256=sha(pp),source_sha256=sourcehash,models_sha256=sha(mp),native_results_sha256=sha(np),forward_count=calls,native_baseline_replays=28,native_target_replays=replays,rows=rows,baselines=baselines,summaries=summaries,macros=macros,files=files),indent=2)+'\n');print('DONE',calls,len(rows),flush=True)
if __name__=='__main__':main()

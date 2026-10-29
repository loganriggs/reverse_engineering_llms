"""M56 common final-position native color-task/format deltas; all frozen candidates."""
from pathlib import Path
import json,hashlib
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def margin(z,g,ep):return float(torch.logsumexp(z.double()[g[ep]],0)-torch.logsumexp(z.double()[g[1-ep]],0))
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 pp=V/'color_causal_availability_protocol.json';protocol=json.loads(pp.read_text());mp=V/'broader_training_screen_manifest.json';assert sha(mp)==protocol['manifest_sha256'];assert sha(V/'broader_training_screen.json')==protocol['competence_sha256'];manifest=json.loads(mp.read_text());modelp=V/'joint_shared_models.json';models=json.loads(modelp.read_text())['models'];assert len(models)==18;op=V/'color_causal_availability.json';assert not op.exists();model,receipt=load_model();mlp=model.transformer.h[1].mlp
 states={}
 for e in models:
  path=Path(e['path']);assert sha(path)==e['sha256'];s=torch.load(path,weights_only=True);f=Featurizer(e['kind'],e['seed']).cuda();f.load_state_dict(s['state']);states[e['kind'],e['seed'],e['arm']]=(f,s['mean'].cuda(),s['scale'].cuda())
 captures={};calls=0;old=json.loads((V/'broader_training_screen.json').read_text());oldpath=Path(old['logits_path']);assert sha(oldpath)==old['logits_sha256'];prior={(x['id'],x['endpoint']):x['logits'] for x in torch.load(oldpath,weights_only=True)}
 for p in manifest['pairs']:
  if p['family']!='object_color':continue
  assert p['valid']
  for ep,e in enumerate(p['endpoints']):
   ids=torch.tensor([e['ids']],device='cuda');out,cap=trace(model,ids);calls+=1;z=out[0,-1];assert torch.equal(z.cpu().double(),prior[p['id'],ep]);captures[p['id'],ep]=dict(write=mlp.Down(cap[1]['products']),logits=z,ids=ids,groups=e['answer_groups'],source=e['source'])
 native=[];records=[];baselines=[];samples=dict(native={},student_base={},student_changed={},directions={},writes={});checks=[]
 for p in manifest['pairs']:
  if p['family']!='object_color' or p['format']!='few_shot_completion':continue
  for ep in [0,1]:
   c=captures[p['id'],ep];key=p['id']+f':endpoint{ep}';base=c['logits'];nlp=base.double().log_softmax(-1);nb=margin(base,c['groups'],ep);samples['native'][key+':base']=base.cpu();samples['writes'][key]=c['write'].cpu();deltas={};targets={}
   for condition,donor in [('task_donor',captures[p['id'],1-ep]),('format_donor',captures[p['id'].replace('few_shot_completion','explicit_question_answer'),ep]),('identity',c)]:
    delta=torch.zeros_like(c['write']);delta[:,-1]=donor['write'][:,-1]-c['write'][:,-1];assert not bool(delta[:,:-1].count_nonzero());assert torch.equal(c['write'][:,-1]+delta[:,-1],donor['write'][:,-1]) or torch.allclose(c['write'][:,-1]+delta[:,-1],donor['write'][:,-1],rtol=1e-6,atol=1e-5);noop=not bool(delta.count_nonzero())
    if noop:out=base
    else:logits,_=trace(model,c['ids'],replace=lambda j,z,w:w+delta if j==1 else w);calls+=1;out=logits[0,-1]
    lp=out.double().log_softmax(-1);response=lp-nlp;effect=margin(out,c['groups'],ep)-nb;energy=float((nlp.exp()*response.square()).sum());dh=sha_tensor(delta);row=dict(key=key,pair_id=p['id'],endpoint=ep,source=c['source'],condition=condition,noop=noop,delta_sha256=dh,delta_energy=float(delta.double().square().sum()),native_response_energy=energy,native_margin_effect=effect,native_margin_energy=effect**2,output_KL=float((nlp.exp()*(nlp-lp)).sum()),native_margin_before=nb,native_top1_before=int(base.argmax()) in c['groups'][ep],native_top1_after=int(out.argmax()) in c['groups'][ep]);native.append(row);targets[condition]=(response,effect,lp,row);deltas[condition]=delta;samples['native'][key+':'+condition]=out.cpu();samples['directions'][key+':'+condition]=dict(delta=delta.cpu(),donor_final=donor['write'][:,-1].cpu(),groups=c['groups'],endpoint=ep)
   for e in models:
    mk=e['kind'],e['seed'],e['arm'];f,mu,scale=states[mk];rec,support,code=f((c['write']-mu)/scale);q=rec*scale+mu;out,_=trace(model,c['ids'],replace=lambda j,z,w:q+mlp.Down_bias if j==1 else w);calls+=1;student=out[0,-1];slp=student.double().log_softmax(-1);sm=margin(student,c['groups'],ep);sk=key+'|'+':'.join(map(str,mk));samples['student_base'][sk]=student.cpu();baselines.append({k:e[k] for k in ['kind','seed','arm']}|dict(key=key,final_KL=float((nlp.exp()*(nlp-slp)).sum()),active_mean=float(support.sum(-1).mean()),task_top1=int(student.argmax()) in c['groups'][ep],binary=sm>0))
    for condition,delta in deltas.items():
     response,effect,lp,nrow=targets[condition]
     if nrow['noop']:out=student
     else:logits,_=trace(model,c['ids'],replace=lambda j,z,w:q+delta+mlp.Down_bias if j==1 else w);calls+=1;out=logits[0,-1]
     change=out.double().log_softmax(-1)-slp;seffect=margin(out,c['groups'],ep)-sm;records.append({k:e[k] for k in ['kind','seed','arm']}|dict(key=key,condition=condition,delta_sha256=nrow['delta_sha256'],native_response_energy=nrow['native_response_energy'],native_margin_energy=nrow['native_margin_energy'],response_error=float((nlp.exp()*(change-response).square()).sum()),margin_error=(seffect-effect)**2,student_margin_effect=seffect));samples['student_changed'][sk+'|'+condition]=out.cpu()
   print('RECIPIENT',len(baselines)//18,'FORWARDS',calls,flush=True)
 summaries=[]
 for e in models:
  for condition in ['task_donor','format_donor','identity']:
   rr=[r for r in records if all(r[k]==e[k] for k in ['kind','seed','arm']) and r['condition']==condition];en=sum(r['native_response_energy'] for r in rr);er=sum(r['response_error'] for r in rr);me=sum(r['native_margin_energy'] for r in rr);mr=sum(r['margin_error'] for r in rr);summaries.append({k:e[k] for k in ['kind','seed','arm']}|dict(condition=condition,n=len(rr),target_energy=en,error_energy=er,relative_response_error=er/en if en else None,margin_target_energy=me,margin_error_energy=mr,relative_margin_error=mr/me if me else None))
 assert len(native)==24 and len(baselines)==144 and len(records)==432;sp=A/'emlp_m56_samples.pt';assert not sp.exists();torch.save(samples,sp);op.write_text(json.dumps(dict(status='completed',protocol_sha256=sha(pp),models_sha256=sha(modelp),manifest_sha256=sha(mp),source_sha256=sha(Path(__file__)),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,competence_logit_replays=16,native=native,baselines=baselines,records=records,summaries=summaries,samples_path=str(sp),samples_sha256=sha(sp)),indent=2)+'\n');print('DONE',calls,flush=True)
def sha_tensor(t):return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()
if __name__=='__main__':main()

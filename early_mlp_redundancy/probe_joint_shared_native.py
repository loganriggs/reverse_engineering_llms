"""M37 identical external native directions across all three student architectures."""
from pathlib import Path
import hashlib,json,math
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def kl(a,b):return float((a.exp()*(a-b)).sum())
def summarize(records,models):
 summaries=[]
 for m in models:
  for region in ['prefix','task']:
   for generator in ['all','scalar','block']:
    base=[x for x in records if all(x[k]==m[k] for k in ['kind','seed','arm']) and x['region']==region and (generator=='all' or x['generator']==generator)]
    strata=[('all',None)]+[(q,None) for q in ['evaluation:train','train:evaluation','evaluation:evaluation']]+[('all',f) for f in sorted({x['family'] for x in base})]
    for quadrant,family in strata:
     rs=[x for x in base if (quadrant=='all' or x['lexical_split']+':'+x['template_split']==quadrant) and (family is None or x['family']==family)]
     me=sum(x['margin_squared_error'] for x in rs);mn=sum(x['native_margin_energy'] for x in rs);le=sum(x['logprob_response_error'] for x in rs);ln=sum(x['native_logprob_response_energy'] for x in rs)
     summaries.append({k:m[k] for k in ['kind','seed','arm']}|dict(region=region,generator=generator,quadrant=quadrant,family=family,interventions=len(rs),nonzero=sum(not x['noop'] for x in rs),margin_error_sum=me,native_margin_energy_sum=mn,relative_margin_error=me/mn if mn else None,logprob_response_error_sum=le,native_logprob_response_energy_sum=ln,relative_logprob_response_error=le/ln if ln else None))
 return summaries
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');mp=P/'joint_shared_models.json';manifest=json.loads(mp.read_text());assert json.loads((P/'joint_shared_training_audit.json').read_text())['results_sha256']==manifest['training_sha256'];bp=P/'context_training_protocol.json';benchmark=json.loads(bp.read_text());oldp=P/'context_native_fidelity.json';old=json.loads(oldp.read_text());assert old['protocol_sha256']==sha(bp);prior={(x['pair_id'],x['endpoint'],x['kind'],x['seed'],x['region'],x['group']):x for x in old['records'] if x['arm']=='initial'};expanded={x['id']:x for x in json.loads((P/'context_expansion_protocol.json').read_text())['rows']};op=P/'joint_shared_native_fidelity.json';assert not op.exists()
 model,receipt=load_model();model.requires_grad_(False);device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;fs={};states={}
 def load(entry):
  assert sha(Path(entry['path']))==entry['sha256'];s=torch.load(entry['path'],weights_only=True);f=Featurizer(entry['kind'],entry['seed']).to(device);f.load_state_dict(s['state']);return f,s['mean'].to(device),s['scale'].to(device)
 for entry in manifest['models']:fs[entry['kind'],entry['seed'],entry['arm']]=load(entry)
 basis={(x['kind'],x['seed']):load(x) for x in benchmark['initial_models']}
 records=[];baselines=[];samples=[];calls=0;noops=0;completed=0;progress=A/'emlp_m37_native_progress.pt'
 if progress.exists():
  saved=torch.load(progress,weights_only=True);assert saved['models_sha256']==sha(mp) and saved['benchmark_sha256']==sha(bp);records,baselines,samples,calls,noops,completed=[saved[k] for k in ['records','baselines','samples','calls','noops','completed']]
 for pi,selected in enumerate(benchmark['causal_pairs']):
  if pi<completed:continue
  pair=expanded[selected['pair_id']];boundary=pair['prefix_length']
  for ep in [0,1]:
   ids=torch.tensor([pair['token_ids'][ep]],device=device);native,cap=trace(model,ids);calls+=1;write=mlp.Down(cap[1]['products']);native=native[0,-1];nlp=native.double().log_softmax(-1);groups=pair['answer_groups'][ep];variant=pair['variants'][ep];nt=competence(native,groups,variant)
   for seed in [71,72]:
    intacts={}
    for entry in [x for x in manifest['models'] if x['seed']==seed]:
     key=entry['kind'],seed,entry['arm'];f,mu,scale=fs[key];rec,support,_=f((write-mu)/scale);q=rec*scale+mu;out,_=trace(model,ids,replace=lambda j,z,w:q+mlp.Down_bias if j==1 else w);calls+=1;logits=out[0,-1];lp=logits.double().log_softmax(-1);task=competence(logits,groups,variant);intacts[key]=(q,logits,lp,task)
     baselines.append(dict(pair_id=pair['id'],endpoint=ep,kind=key[0],seed=seed,arm=key[2],family=pair['family'],lexical_split=pair['lexical_split'],template_split=pair['template_split'],native_task=nt,task=task,final_KL=kl(nlp,lp),active_mean=float(support.sum(-1).mean())))
    for generator in ['scalar','block']:
     f0,mu0,scale0=basis[generator,seed];_,support0,code=f0((write-mu0)/scale0);D=torch.nn.functional.normalize(f0.dictionary,dim=0)
     for spec in [x for x in benchmark['causal_interventions'] if x['pair_id']==pair['id'] and x['endpoint']==ep and x['seed']==seed]:
      mask=torch.zeros_like(write[:,:,:1]);mask[:,:boundary]=int(spec['region']=='prefix');mask[:,boundary:]=int(spec['region']=='task')
      for gi,g in enumerate(spec['groups']):
       atoms=torch.arange(g*4,g*4+4,device=device);delta=(code[:,:,atoms]@D[:,atoms].T)*scale0*mask;noop=not bool(torch.count_nonzero(delta));noops+=noop
       if noop:nchanged=native
       else:
        out,_=trace(model,ids,replace=lambda j,z,w:w-delta if j==1 else w);calls+=1;nchanged=out[0,-1]
       nclp=nchanged.double().log_softmax(-1);nc=competence(nchanged,groups,variant);neffect=nt['margin']-nc['margin'];nresponse=nclp-nlp;energy=float((nlp.exp()*nresponse.square()).sum());de=float(delta.double().square().sum());oldrow=prior[pair['id'],ep,generator,seed,spec['region'],g]
       assert abs(oldrow['native_margin_effect']-neffect)<1e-10 and abs(oldrow['native_logprob_response_energy']-energy)<1e-10 and oldrow['noop']==noop and abs(oldrow['delta_energy']-de)<1e-8
       dh=hashlib.sha256(delta.cpu().numpy().tobytes()).hexdigest()
       for key,(q,logits,lp,task) in intacts.items():
        if noop:changed=logits
        else:
         out,_=trace(model,ids,replace=lambda j,z,w:q-delta+mlp.Down_bias if j==1 else w);calls+=1;changed=out[0,-1]
        clp=changed.double().log_softmax(-1);ct=competence(changed,groups,variant);effect=task['margin']-ct['margin'];error=(clp-lp)-nresponse
        row=dict(pair_id=pair['id'],endpoint=ep,kind=key[0],seed=seed,arm=key[2],generator=generator,family=pair['family'],lexical_split=pair['lexical_split'],template_split=pair['template_split'],region=spec['region'],group=g,noop=noop,delta_sha256=dh,delta_energy=de,native_margin_effect=neffect,reconstructed_margin_effect=effect,margin_squared_error=(effect-neffect)**2,native_margin_energy=neffect**2,logprob_response_error=float((nlp.exp()*error.square()).sum()),native_logprob_response_energy=energy,intact_final_KL=kl(nlp,lp),perturbed_final_KL=kl(nclp,clp));records.append(row)
        if pi in [0,6,16] and ep==0 and gi==0:samples.append(dict(row=len(records)-1,native=native.cpu(),native_changed=nchanged.cpu(),reconstructed=logits.cpu(),reconstructed_changed=changed.cpu(),groups=groups,variant=variant,delta=delta.cpu(),code=code.cpu(),D=D.cpu(),atoms=atoms.cpu(),scale=scale0.cpu(),mask=mask.cpu()))
  tmp=progress.with_suffix('.tmp');torch.save(dict(models_sha256=sha(mp),benchmark_sha256=sha(bp),records=records,baselines=baselines,samples=samples,calls=calls,noops=noops,completed=pi+1),tmp);tmp.replace(progress);print('PAIRS',pi+1,len(benchmark['causal_pairs']),'FORWARDS',calls,flush=True)
 assert len(records)==6336 and len(baselines)==792 and noops==96
 sp=A/'emlp_m37_native_samples.pt';torch.save(samples,sp);result=dict(models_sha256=sha(mp),benchmark_sha256=sha(bp),benchmark_results_sha256=sha(oldp),source_sha256=sha(Path(__file__)),checkpoint_sha256=receipt.weights_sha256,records=records,baselines=baselines,summaries=summarize(records,manifest['models']),forward_count=calls,noop_directions=noops,samples_path=str(sp),samples_sha256=sha(sp),scope='Both original M18 benchmark generators applied identically to every architecture of its seed, including shared_fixed; familiar development evaluation, not semantic discovery.')
 op.write_text(json.dumps(result,indent=2)+'\n');print('DONE',calls,len(records),flush=True)
if __name__=='__main__':main()

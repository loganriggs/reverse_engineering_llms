"""M29 response-supervised models on the unchanged M18 native benchmark."""
from pathlib import Path
import hashlib,json
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def kl(lp,lq):return float((lp.exp()*(lp-lq)).sum())
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=P/'basis_transfer_protocol.json';p=json.loads(pp.read_text());bp=P/'m18_basis_benchmark_protocol.json';benchmark=json.loads(bp.read_text());oldpath=P/'context_native_fidelity.json';old=json.loads(oldpath.read_text());assert old['protocol_sha256']==benchmark['parent_protocol_sha256']==sha(P/'context_training_protocol.json');oldrecords={(x['pair_id'],x['endpoint'],x['kind'],x['seed'],x['region'],x['group']):x for x in old['records'] if x['arm']=='initial'};rp=P/'basis_transfer_models.json';r=json.loads(rp.read_text());assert sha(pp)==r['protocol_sha256'];assert json.loads((P/'basis_transfer_models_audit.json').read_text())['result_sha256']==sha(rp);outpath=P/'basis_transfer_native_fidelity.json';assert not outpath.exists();expanded={x['id']:x for x in json.loads((P/'context_expansion_protocol.json').read_text())['rows']};model,receipt=load_model();model.requires_grad_(False);device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;featurizers={};states={}
 for entry in r['models']:
  path=Path(entry['path']);assert sha(path)==entry['sha256'];s=torch.load(path,weights_only=True);f=Featurizer(entry['kind'],entry['seed']).to(device);f.load_state_dict(s['state']);key=entry['kind'],entry['seed'],entry['arm'];featurizers[key]=f;states[key]=(s['mean'].to(device),s['scale'].to(device))
 basis={}
 for entry in benchmark['initial_models']:
  path=Path(entry['path']);assert sha(path)==entry['sha256'];state=torch.load(path,weights_only=True);f=Featurizer(entry['kind'],entry['seed']).to(device);f.load_state_dict(state['state']);basis[entry['kind'],entry['seed']]=(f,state['mean'].to(device),state['scale'].to(device))
 records=[];baselines=[];samples=[];calls=0;noop_directions=0;completed=0;progress=A/'emlp_m29_basis_progress.pt'
 if progress.exists():
  prior=torch.load(progress,weights_only=True);assert prior['protocol_sha256']==sha(pp);records=prior['records'];baselines=prior['baselines'];samples=prior['samples'];calls=prior['calls'];noop_directions=prior['noop_directions'];completed=prior['completed'];print('RESUME',completed,flush=True)
 for pi,selected in enumerate(benchmark['causal_pairs']):
  if pi<completed:continue
  pair=expanded[selected['pair_id']];boundary=pair['prefix_length']
  for ep in [0,1]:
   ids=torch.tensor([pair['token_ids'][ep]],device=device);native,cap=trace(model,ids);calls+=1;v=mlp.Down(cap[1]['products']);native=native[0,-1];nlp=native.double().log_softmax(-1);groups=pair['answer_groups'][ep];variant=pair['variants'][ep];ntask=competence(native,groups,variant)
   for initial in p['initial_models']:
    kind,seed=initial['kind'],initial['seed'];key=kind,seed,'initial';f0,mu,scale=basis[kind,seed];pred,support,code=f0((v-mu)/scale);D=torch.nn.functional.normalize(f0.dictionary,dim=0);intacts={}
    for arm in ['initial']+p['arms']:
     f=featurizers[kind,seed,arm];mm,ss=states[kind,seed,arm];assert torch.equal(mm,mu) and torch.equal(ss,scale);rec,_,_=f((v-mu)/scale);q=rec*scale+mu;out,_=trace(model,ids,replace=lambda i,z,w:q+mlp.Down_bias if i==1 else w);calls+=1;lg=out[0,-1];lp=lg.double().log_softmax(-1);task=competence(lg,groups,variant);intacts[arm]=(q,lg,lp,task);baselines.append(dict(pair_id=pair['id'],endpoint=ep,kind=kind,seed=seed,arm=arm,family=pair['family'],lexical_split=pair['lexical_split'],template_split=pair['template_split'],native_task=ntask,task=task,final_KL=kl(nlp,lp)))
    specs=[x for x in benchmark['causal_interventions'] if x['pair_id']==pair['id'] and x['endpoint']==ep and x['seed']==seed]
    for spec in specs:
     mask=torch.zeros_like(v[:,:,:1]);mask[:,:boundary]=1 if spec['region']=='prefix' else 0;mask[:,boundary:]=1 if spec['region']=='task' else 0
     for gi,g in enumerate(spec['groups']):
      atoms=torch.arange(g*4,g*4+4,device=device);delta=(code[:,:,atoms]@D[:,atoms].T)*scale*mask;noop=bool((delta==0).all());noop_directions+=int(noop)
      if noop:native_changed=native
      else:
       output,_=trace(model,ids,replace=lambda i,z,w:w-delta if i==1 else w);calls+=1;native_changed=output[0,-1]
      nclp=native_changed.double().log_softmax(-1);nc=competence(native_changed,groups,variant);neffect=ntask['margin']-nc['margin'];nresponse=nclp-nlp;weight=nlp.exp();response_energy=float((weight*nresponse.square()).sum())
      for arm,(q,lg,lp,task) in intacts.items():
       if noop:changed=lg
       else:
        output,_=trace(model,ids,replace=lambda i,z,w:q-delta+mlp.Down_bias if i==1 else w);calls+=1;changed=output[0,-1]
       clp=changed.double().log_softmax(-1);ct=competence(changed,groups,variant);effect=task['margin']-ct['margin'];discrepancy=(clp-lp)-nresponse
       row=dict(pair_id=pair['id'],endpoint=ep,kind=kind,seed=seed,arm=arm,family=pair['family'],lexical_split=pair['lexical_split'],template_split=pair['template_split'],region=spec['region'],group=g,noop=noop,delta_energy=float(delta.double().square().sum()),native_margin_effect=neffect,reconstructed_margin_effect=effect,margin_squared_error=(effect-neffect)**2,native_margin_energy=neffect**2,logprob_response_error=float((weight*discrepancy.square()).sum()),native_logprob_response_energy=response_energy,intact_final_KL=kl(nlp,lp),perturbed_final_KL=kl(nclp,clp));records.append(row)
       if pi in [0,6,16] and ep==0 and gi==0:samples.append(dict(row=len(records)-1,native=native.cpu(),native_changed=native_changed.cpu(),reconstructed=lg.cpu(),reconstructed_changed=changed.cpu(),groups=groups,variant=variant,write=v.cpu(),support=support.cpu(),mean=mu.cpu(),code=code.cpu(),D=D.cpu(),atoms=atoms.cpu(),scale=scale.cpu(),mask=mask.cpu(),delta=delta.cpu(),ids=ids.cpu(),prefix_length=boundary))
  print('CAUSAL_PAIRS',pi+1,len(benchmark['causal_pairs']),'FORWARDS',calls,flush=True)
  tmp=progress.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(pp),records=records,baselines=baselines,samples=samples,calls=calls,noop_directions=noop_directions,completed=pi+1),tmp);tmp.replace(progress)
 assert len(records)==3520 and len(baselines)==880
 summaries=[]
 for kind in ['scalar','block']:
  for seed in [71,72]:
   for arm in ['initial']+p['arms']:
    for region in ['prefix','task']:
     base=[x for x in records if (x['kind'],x['seed'],x['arm'],x['region'])==(kind,seed,arm,region)]
     strata=[('all',None)]+[(q,None) for q in ['evaluation:train','train:evaluation','evaluation:evaluation']]+[('all',f) for f in sorted({x['family'] for x in base})]
     for quadrant,family in strata:
      rr=[x for x in base if (quadrant=='all' or x['lexical_split']+':'+x['template_split']==quadrant) and (family is None or x['family']==family)];me=sum(x['margin_squared_error'] for x in rr);mn=sum(x['native_margin_energy'] for x in rr);le=sum(x['logprob_response_error'] for x in rr);ln=sum(x['native_logprob_response_energy'] for x in rr);summaries.append(dict(kind=kind,seed=seed,arm=arm,region=region,quadrant=quadrant,family=family,interventions=len(rr),nonzero=sum(not x['noop'] for x in rr),margin_error_sum=me,native_margin_energy_sum=mn,relative_margin_error=me/mn if mn else None,margin_RMSE=(me/len(rr))**.5,logprob_response_error_sum=le,native_logprob_response_energy_sum=ln,relative_logprob_response_error=le/ln if ln else None,intact_final_KL=sum(x['intact_final_KL'] for x in rr)/len(rr),perturbed_final_KL=sum(x['perturbed_final_KL'] for x in rr)/len(rr)))
 ap=A/'emlp_m29_basis_transfer_native_samples.pt';torch.save(samples,ap);out=dict(status='completed',protocol_sha256=sha(pp),benchmark_protocol_sha256=sha(bp),benchmark_results_sha256=sha(oldpath),training_results_sha256=sha(rp),source_sha256=sha(Path(__file__)),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,noop_directions=noop_directions,records=records,baselines=baselines,summaries=summaries,samples_path=str(ap),samples_sha256=sha(ap));outpath.write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls,'NOOP_DIRECTIONS',noop_directions,flush=True)
if __name__=='__main__':main()

"""M16 fixed prefix/task localization of frozen context-role ablations."""
import hashlib,json,sys
import tiktoken
from pathlib import Path
import numpy as np
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from probe_mlp1_group_deletion import metrics
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent;A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');cp=P/'validation/context_challenge_protocol.json';challenge=json.loads(cp.read_text());screenpath=P/'validation/context_competence.json';screen=json.loads(screenpath.read_text());assert sha(cp)==screen['protocol_sha256'];passed={(x['context'],x['family']) for x in screen['cohorts'] if x['pass_gate']};rows={}
 for pair in challenge['rows']:
  if (pair['context'],pair['family']) in passed:
   key=pair['context']+'|'+pair['id'];rows[key]=dict(pair,base_id=pair['id'])
 ids0=list(rows);entries=challenge['models'];enc=tiktoken.get_encoding('gpt2');bank={x['id']:x for x in json.loads((P/'validation/training_source_context_audit.json').read_text())['rows']};boundaries={}
 for pid,pair in rows.items():
  prefix=enc.encode(challenge['prefixes'][pair['context']]);n=len(prefix);assert n>0
  for ep in [0,1]:
   assert pair['token_ids'][ep]==prefix+bank[pair['base_id']]['token_ids'][ep], 'Tokenization crosses prefix/task boundary'
   assert pair['texts'][ep]==challenge['prefixes'][pair['context']]+bank[pair['base_id']]['texts'][ep]
  boundaries[pid]=n
 oldpath=P/'validation/context_group_roles.json';old=json.loads(oldpath.read_text());oldbaselines={(x['id'],x['endpoint'],x['kind'],x['seed']):x for x in old['baselines']}
 protocol=dict(challenge_sha256=sha(cp),screen_sha256=sha(screenpath),matching_sha256=challenge['matching_sha256'],all_position_reference_sha256=sha(oldpath),source_sha256=sha(Path(__file__)),selected_pair_keys=ids0,prefix_lengths=boundaries,regions=['prefix','task'],design='All32groups, allpassingcohorts, samefixedmodels/assignments/gains, allscalar/shuffledcontrols. Remove decoded contribution only in specified token region; fixedothercodes/gates/biases. Compare final answer margin and final KL to intact reconstruction. Interaction=all-position margin loss minus prefix loss minus task loss. KLs are not additive. No new fitting/selection. Region comparison is localization, not semantic-vs-position identification.',stopping_rule='One fixed localization panel. No gain/clipping/matching/model search. Use to define causal supervision sites; weak task-region correspondence requires broader contexts before retraining or unit claims.',scope='Reused M15 development contexts and native gate; not new tasks, new lexical transfer, or manifold validation.')
 pp=P/'validation/context_region_protocol.json'
 if pp.exists():assert json.loads(pp.read_text())==protocol, 'Frozen protocol mismatch'
 else:pp.write_text(json.dumps(protocol,indent=2)+'\n')
 if '--prepare-only' in sys.argv:print('Protocol frozen; verified',len(rows)*2,'token boundaries',flush=True);return
 assert not (P/'validation/context_region_roles.json').exists(), 'Completed results already exist'
 model,receipt=load_model();device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;models=[]
 for entry in entries:
  path=Path(entry['path']);assert sha(path)==entry['sha256'];s=torch.load(path,weights_only=True);f=Featurizer(entry['kind'],entry['seed']).to(device);f.load_state_dict(s['state']);D=torch.nn.functional.normalize(f.dictionary,dim=0);groups={'contiguous':np.arange(128).reshape(32,4)}
  if entry['kind']=='block':groups['shuffled']=np.random.default_rng(1200+entry['seed']).permutation(128).reshape(32,4)
  models.append((entry,f,D,s['mean'].to(device),s['scale'].to(device),groups))
 records=[];baselines=[];samples=[];calls=0;noops=0
 for pi,pid in enumerate(ids0):
  pair=rows[pid];part='evaluation'
  for ep in [0,1]:
   ids=torch.tensor([pair['token_ids'][ep]],device=device);native,cap=trace(model,ids);calls+=1;v=mlp.Down(cap[1]['products']);groups=pair['answer_groups'][ep];variant=int(pair['variants'][ep])
   for entry,f,D,mu,scale,groupings in models:
    reconstruction,support,code=f((v-mu)/scale);q=reconstruction*scale+mu;intact,_=trace(model,ids,replace=lambda i,z,w:q+mlp.Down_bias if i==1 else w);calls+=1;base=competence(intact[0,-1],groups,variant);prior=oldbaselines[(pid,ep,entry['kind'],entry['seed'])];assert abs(base['margin']-prior['task']['margin'])<1e-8;baselines.append(dict(id=pid,context=pair['context'],base_id=pair['base_id'],endpoint=ep,kind=entry['kind'],seed=entry['seed'],split=part,family=pair['family'],task=base,native_task=competence(native[0,-1],groups,variant),**metrics(native,intact,ids)))
    for grouping,indices in groupings.items():
     for g,atoms0 in enumerate(indices):
      for region in ['prefix','task']:
       atoms=torch.tensor(atoms0,device=device);full_contribution=(code[:,:,atoms]@D[:,atoms].T)*scale;mask=torch.zeros_like(q[:,:,:1]);boundary=boundaries[pid];mask[:,:boundary]=1 if region=='prefix' else 0;mask[:,boundary:]=1 if region=='task' else 0;contribution=full_contribution*mask;changed=q-contribution;active=float(((support[:,:,atoms]>0).double()*mask).sum(-1).mean());noop=bool((contribution==0).all())
       if noop:out=intact;noops+=1
       else:out,_=trace(model,ids,replace=lambda i,z,w:changed+mlp.Down_bias if i==1 else w);calls+=1
       task=competence(out[0,-1],groups,variant);row=dict(region=region,prefix_length=boundary,id=pid,context=pair['context'],base_id=pair['base_id'],endpoint=ep,family=pair['family'],split=part,kind=entry['kind'],seed=entry['seed'],grouping=grouping,group=g,noop=noop,mean_removed_active_coordinates=active,removed_relative_energy=float(contribution.double().square().sum()/q.double().square().sum()),margin_loss=base['margin']-task['margin'],intact_task=base,ablated_task=task,**metrics(intact,out,ids));records.append(row)
       if (pi in [0,len(ids0)//2]) and ep==0 and g==0:samples.append(dict(row=len(records)-1,mask=mask.cpu(),intact=intact.cpu(),ablated=out.cpu(),ids=ids.cpu(),groups=groups,variant=variant,code=code.cpu(),D=D.cpu(),atoms=atoms.cpu(),scale=scale.cpu(),q=q.cpu(),contribution=contribution.cpu(),changed=changed.cpu()))
  print('PAIRS',pi+1,len(ids0),'FORWARDS',calls,flush=True)
 ap=A/'emlp_m16_context_region_samples.pt';torch.save(samples,ap);out=dict(status='completed',protocol_sha256=sha(pp),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,noop_interventions=noops,records=records,baselines=baselines,samples_path=str(ap),samples_sha256=sha(ap));(P/'validation/context_region_roles.json').write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls,'NOOPS',noops,flush=True)
if __name__=='__main__':main()

"""H1048 ten-fold whole-family exclusion of encoder and action-decoder training."""
from pathlib import Path
import sys,json,hashlib,ast,time,numpy as np,torch
from torch import nn
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_action_dictionary_family_crossfit';out=A/'action_dictionary_family_crossfit';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');seed=10331
files=[A/'training_coverage_selector_comparison/panel.json',A/'multitask_boundary_contract/states.pt',A/'diverse_task_expansion_screen/states.pt',R/'scripts/run_bilin18_finite_supervised_selectors.py',R/'scripts/run_bilin18_continuous_action_dictionary.py'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};allpairs=json.loads(files[0].read_text())['train'];old=torch.load(files[1],weights_only=False);added=torch.load(files[2],weights_only=False);states={};examples={}
for p in allpairs:
 for i in p['indices']:
  prefix,idx=i.split(':');data=old if prefix=='old' else added;states[i]=tuple(v.cuda() for v in data['states'][int(idx)]);examples[i]=data['examples'][int(idx)]
families=sorted({p['family'] for p in allpairs});assert len(families)==10 and len(allpairs)==80
config=dict(experiment='H1048',seed=seed,folds=families,training='Leave one entire named family out from PCA,normalization,and decoder. Remaining9families,16visits each,144fixed steps. All eligible pairs retained including native errors.',architecture='Same H1047 continuous normalized action decoder,442368parameters, fixed support energy encoder and random scalar/block/shared layout, max48active. Refit training-only PCA128perstream in each fold.',optimization='Adam0.001; bidirectional finite pairedKL plus1e-4direction displacement; fixed final checkpoint only.',control='Evaluate initial and final decoder on full training and heldout family. No evaluation-based checkpoint/architecture/hyperparameter selection.',scope='Competence-screened historical task bank crossfit, not pristine external validation. Related families share lexical content/operations. Eachfold must exclude exact heldout prompts from fitted statistics.',source_hashes=hashes)
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n')
# Exact audited architecture definitions, without executing any training driver.
tree=ast.parse(files[3].read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['Selector','suffix','KL']];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(files[3]),'exec'),globals());OriginalSelector=Selector
tree=ast.parse(files[4].read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['Selector','direct','pair_loss']];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(files[4]),'exec'),globals())
counts={'suffix':0};model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
for p in model.parameters():p.requires_grad_(False)
with torch.no_grad():native_train={i:suffix(st).detach() for i,st in states.items()}
@torch.no_grad()
def assess(selector,pp,checkpoint,split,fold):
 records=[];samples=[];maxactive=0
 for p in pp:
  for i,d in [p['indices'],p['indices'][::-1]]:
   delta=tuple(states[d][k]-states[i][k] for k in range(3));change,supp=direct(selector,delta);assert all(torch.all((s==0)|(s==1)) for s in supp);maxactive=max(maxactive,float(torch.stack([s.sum(1) for s in supp]).sum(0).max()));y=suffix(tuple(states[i][k]+change[k] for k in range(3)));base=native_train[i].double();don=native_train[d].double();value=float(KL(don,y.double()));den=float(KL(don,base));g=examples[i]['groups'];v=examples[d]['variant'];records.append(dict(fold=fold,kind=selector.kind,checkpoint=checkpoint,split=split,family=p['family'],pair=p['id'],row=i,donor=d,paired_KL=value,paired_denominator=den,donor_top1=int(y.argmax()) in g[v]))
   if split=='heldout' and i==pp[0]['indices'][0]:samples.append(dict(fold=fold,kind=selector.kind,checkpoint=checkpoint,row=i,base=base.cpu(),donor=don.cpu(),paired=y.double().cpu()))
 return records,samples,maxactive
allrecords=[];allsamples=[];fitaudits=[];overallactive=0;orth=0
for fold in families:
 trainpairs=[p for p in allpairs if p['family']!=fold];heldpairs=[p for p in allpairs if p['family']==fold];tf=sorted({p['family'] for p in trainpairs});assert len(tf)==9 and fold not in tf
 traintexts={examples[i]['text'] for p in trainpairs for i in p['indices']};heldtexts={examples[i]['text'] for p in heldpairs for i in p['indices']};assert not traintexts&heldtexts
 X=[[],[],[]];weights=[]
 for f in tf:
  ii=sorted({i for p in trainpairs if p['family']==f for i in p['indices']})
  for i in ii:
   T=states[i][0].shape[1]
   for k in range(3):X[k].append(states[i][k].reshape(T,1152).double())
   weights.extend([1/(9*len(ii)*T)]*T)
 w=torch.tensor(weights,dtype=torch.float64,device='cuda');assert abs(float(w.sum())-1)<1e-12;B=[];centers=[];scales=[]
 for k in range(3):
  x=torch.cat(X[k]);mu=(x*w[:,None]).sum(0,keepdim=True);z=x-mu;sc=((z.square()*w[:,None]).sum()/1152).sqrt();ev,b=torch.linalg.eigh((z*w[:,None]).T@z);b=b[:,-128:].flip(1);orth=max(orth,float((b.T@b-torch.eye(128,device='cuda')).abs().max()));B.append(b.float());centers.append(mu.float());scales.append(sc.float().reshape(1,1))
 scale=torch.stack(scales);initial_decoder=F.normalize(torch.stack(B),dim=1);folder=out/fold;folder.mkdir(exist_ok=True);torch.save(dict(bases=[b.cpu() for b in B],scale=scale.cpu(),center=torch.stack(centers).cpu(),training_families=tf),folder/'geometry.pt')
 schedule=[]
 for step in range(144):
  f=tf[step%9];pp=[p for p in trainpairs if p['family']==f];schedule.append(pp[(step//9)%len(pp)])
 assert {p['id'] for p in schedule}=={p['id'] for p in trainpairs}
 foldrecords=[];foldsamples=[];foldfits=[]
 for kind in ['scalar','block','shared']:
  selector=Selector(kind);initial=None;final=None
  for split,pp in [('train',trainpairs),('heldout',heldpairs)]:
   rr,ss,active=assess(selector,pp,'initial',split,fold);foldrecords+=rr;foldsamples+=ss;overallactive=max(overallactive,active)
  opt=torch.optim.Adam(selector.parameters(),lr=.001);history=[];maxgrad=0
  for step,p in enumerate(schedule):
   value=pair_loss(selector,p);loss=value+1e-4*(selector.decoder()-initial_decoder).square().sum(1).mean();opt.zero_grad();loss.backward();assert torch.isfinite(loss) and torch.isfinite(selector.dictionary.grad).all();maxgrad=max(maxgrad,float(selector.dictionary.grad.abs().max()));opt.step()
   if step%48==0:history.append(dict(step=step,family=p['family'],paired_KL=float(value.detach())))
  for split,pp in [('train',trainpairs),('heldout',heldpairs)]:
   rr,ss,active=assess(selector,pp,'final',split,fold);foldrecords+=rr;foldsamples+=ss;overallactive=max(overallactive,active)
  torch.save(dict(state=selector.state_dict(),optimizer=opt.state_dict(),history=history),folder/f'{kind}.pt');foldfits.append(dict(fold=fold,kind=kind,steps=144,training_pairs=len(trainpairs),heldout_pairs=len(heldpairs),trained_families=tf,max_gradient=maxgrad));print('DONE',fold,kind,flush=True)
 (folder/'results.json').write_text(json.dumps(dict(records=foldrecords,fits=foldfits,source_hashes=hashes),indent=2)+'\n');torch.save(foldsamples,folder/'audit_logits.pt');allrecords+=foldrecords;allsamples+=foldsamples;fitaudits+=foldfits
summary=[]
for fold in families:
 for kind in ['scalar','block','shared']:
  for checkpoint in ['initial','final']:
   rr=[r for r in allrecords if r['fold']==fold and r['kind']==kind and r['checkpoint']==checkpoint];tr=[r for r in rr if r['split']=='train'];he=[r for r in rr if r['split']=='heldout'];summary.append(dict(fold=fold,kind=kind,checkpoint=checkpoint,training_KL=float(np.mean([np.mean([r['paired_KL'] for r in tr if r['family']==f]) for f in sorted({r['family'] for r in tr})])),heldout_relative_KL=sum(r['paired_KL'] for r in he)/sum(r['paired_denominator'] for r in he),heldout_donor_top1=float(np.mean([r['donor_top1'] for r in he]))))
audit=dict(source_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),folds=10,fits=len(fitaudits),max_active=overallactive,PCA_orthogonality_max=orth,suffix_sequences=counts['suffix'],training_student_suffix_sequences=30*144*2,heldout_training_prompt_overlap=0,heldout_families_in_fitted_statistics=0,finite=all(np.isfinite(r['paired_KL']) for r in allrecords));assert audit['source_hashes_unchanged'] and audit['finite'] and overallactive<=48 and orth<1e-10
torch.save(allsamples,out/'audit_logits.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=allrecords,fits=fitaudits,audits=audit),indent=2)+'\n');print(json.dumps(audit,indent=2))

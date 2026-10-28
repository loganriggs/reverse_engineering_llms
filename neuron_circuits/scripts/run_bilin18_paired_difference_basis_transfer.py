"""H1037 fixed-rank paired-difference versus activation geometry."""
from pathlib import Path
import sys,json,hashlib,ast,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_paired_difference_basis_transfer';out=A/'paired_difference_basis_transfer';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
import run_subject_number_sparse_graph_token_extraction_v1 as runner
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
train=torch.load(A/'multitask_boundary_contract/states.pt',weights_only=False);tp=json.loads((A/'multitask_boundary_contract/panel.json').read_text());families=sorted({p['family'] for p in tp});assert len(tp)==60 and len(families)==5
held=torch.load(A/'selector_task_holdouts/states.pt',weights_only=False);panel=json.loads((A/'selector_task_holdouts/panel.json').read_text());old=json.loads((A/'bilin18_selector_task_holdouts.json').read_text());gates={(r['family'],r['split']) for r in old['competence'] if r['pass_gate']};pairs=[r for r in panel if (r['family'],r['split']) in gates];ix=sorted({i for p in pairs for i in p['indices']})
assert not {e['text'] for e in train['examples'].values()}&{e['text'] for e in held['examples'].values()}
files=[A/'multitask_boundary_contract/states.pt',A/'multitask_boundary_contract/panel.json',A/'multistream_span_diagnosis/pca.pt',A/'selector_task_holdouts/states.pt'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
config=dict(experiment='H1037',rank=128,training='60 discovery pairs, five families. Family/pair/token-balanced uncentered second moment of donor-minus-recipient residual. Equivalent to covariance of both signed orientations. No downstream loss used.',arms=['task','paired_difference'],source_hashes=hashes,other_ports='Native; paired interventions transfer exact donor other ports, matching H1035.',evaluation='Same H1035 competent historical whole-task exclusions; opened diagnostic panel, not fresh validation.',center='Both bases use identical discovery activation center for whole-state reconstruction. Center cancels in paired changes.',limits='Task-change covariance is supervision by paired data, not finite causal-loss training. Same rank, not same training statistic.',fit_holdout_rows=0)
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n')
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
# Reuse only pure suffix/metric definitions from the audited fixed factorial.
tree=ast.parse((R/'scripts/run_bilin18_projection_stream_factorial.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['suffix','KL','margin']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<audited-functions>','exec'),globals())
with torch.no_grad():
 pca=torch.load(A/'multistream_span_diagnosis/pca.pt',weights_only=False);mu=pca['center'][0].float().cuda();fits={'task':(mu,pca['bases'][0].float().cuda())};counts={f:sum(p['family']==f for p in tp) for f in families};D=[];w=[]
 for p in tp:
  i,d=p['indices'];delta=(train['states'][d][0]-train['states'][i][0]).reshape(-1,1152).double();D.append(delta);w.extend([1/(len(families)*counts[p['family']]*len(delta))]*len(delta))
 D=torch.cat(D).cuda();w=torch.tensor(w,dtype=torch.float64,device='cuda');assert abs(float(w.sum())-1)<1e-12
 cov=(D*w[:,None]).T@D;ev,b=torch.linalg.eigh(cov);b=b[:,-128:].flip(1);orth=float((b.T@b-torch.eye(128,device='cuda')).abs().max());fits['paired_difference']=(mu,b.float());rank=int((ev>ev[-1]*1e-10).sum());assert rank>=128
 train_retained={name:float(((D@basis.double())**2*w[:,None]).sum()/(D.square()*w[:,None]).sum()) for name,(_,basis) in fits.items()}
 states={i:tuple(x.cuda() for x in held['states'][i]) for i in ix};native={i:held['native'][i].cuda() for i in ix};records=[];samples=[];energies=[]
 for arm,(mu,b) in fits.items():
  pred={}
  for i in ix:
   raw=states[i][0];z=raw.reshape(-1,1152)-mu;pz=(z@b)@b.T;pred[i]=(pz+mu).reshape_as(raw);energies.append(dict(arm=arm,row=i,relative_residual_MSE=float((pz-z).square().sum()/z.square().sum())))
  for p in pairs:
   for i,d in [p['indices'],p['indices'][::-1]]:
    y=suffix((pred[i],*states[i][1:]));yp=suffix((states[i][0]+pred[d]-pred[i],states[d][1],states[d][2]));base=native[i];don=native[d];g=held['examples'][i]['groups'];end=p['indices'].index(i)
    records.append(dict(arm=arm,family=p['family'],split=p['split'],row=i,donor=d,whole_KL=KL(base,y),paired_KL=KL(don,yp),paired_denominator=KL(don,base),correct_top1=int(y.argmax()) in g[end]))
    if i==pairs[0]['indices'][0]:samples.append(dict(arm=arm,row=i,base=base.cpu(),donor=don.cpu(),whole=y.cpu(),paired=yp.cpu()))
  print('EVALUATED',arm,flush=True)
summary=[]
for arm in fits:
 for f,sp in sorted(gates):
  rr=[r for r in records if (r['arm'],r['family'],r['split'])==(arm,f,sp)];summary.append(dict(arm=arm,family=f,split=sp,whole_KL=float(np.mean([r['whole_KL'] for r in rr])),paired_relative_KL=sum(r['paired_KL'] for r in rr)/sum(r['paired_denominator'] for r in rr),top1=float(np.mean([r['correct_top1'] for r in rr]))))
previous=json.loads((A/'bilin18_projection_stream_factorial.json').read_text());cross=max(abs(s[k]-next(r for r in previous['summary'] if r['arm']=='100' and r['family']==s['family'] and r['split']==s['split'])[k]) for s in summary if s['arm']=='task' for k in ['whole_KL','paired_relative_KL','top1'])
audit=dict(fit_holdout_rows=0,prefix_sequences=0,source_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),training_difference_effective_rank=rank,training_difference_retained_energy=train_retained,suffix_sequences=len(records)*2,PCA_orthogonality_max=orth,task_baseline_summary_difference=cross,finite=all(np.isfinite(r['whole_KL']) and np.isfinite(r['paired_KL']) for r in records))
assert orth<1e-10 and cross<1e-4 and audit['finite'] and audit['source_hashes_unchanged']
torch.save({name:dict(center=mu.cpu(),basis=b.cpu()) for name,(mu,b) in fits.items()},out/'fits.pt');torch.save(samples,out/'audit_logits.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,energies=energies,audits=audit),indent=2)+'\n');print(json.dumps(audit,indent=2))

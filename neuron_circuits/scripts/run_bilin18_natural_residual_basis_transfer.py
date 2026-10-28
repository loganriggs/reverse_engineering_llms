"""H1036 fixed-rank training-distribution comparison, preserving native other ports."""
from pathlib import Path
import sys,json,hashlib,ast,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_natural_residual_basis_transfer';out=A/'natural_residual_basis_transfer';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
import run_subject_number_sparse_graph_token_extraction_v1 as runner
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
manifest=json.loads((A/'bilin18_natural_residual_discovery_manifest.json').read_text());docs=manifest['documents'][:192];original=json.loads((A/'bilin18_early_bsf_data_manifest.json').read_text());reserved={r['document_hash'] for r in original['records'] if r['split']!='train'};assert len(docs)==192 and not reserved&{r['document_hash'] for r in docs}
arrays={};tokens=[]
for r in docs:
 if r['source'] not in arrays:arrays[r['source']]=np.load(r['source'],mmap_mode='r')
 ids=np.asarray(arrays[r['source']][r['source_row'],:128],dtype=np.int64);assert hashlib.sha256(ids.tobytes()).hexdigest()==r['window_sha256_int64'];tokens.append(ids[:16].copy())
tokens=torch.tensor(np.stack(tokens));train=torch.load(A/'multitask_boundary_contract/states.pt',weights_only=False);count=sum(train['states'][i][0].shape[1] for i in train['states']);assert count<=3072
held=torch.load(A/'selector_task_holdouts/states.pt',weights_only=False);panel=json.loads((A/'selector_task_holdouts/panel.json').read_text());old=json.loads((A/'bilin18_selector_task_holdouts.json').read_text());gates={(r['family'],r['split']) for r in old['competence'] if r['pass_gate']};pairs=[r for r in panel if (r['family'],r['split']) in gates];ix=sorted({i for p in pairs for i in p['indices']})
# Reject any evaluated exact token sequence occurring in the selected natural prefixes.
overlap=0
for e in held['examples'].values():
 q=e['ids']
 for ids in tokens.tolist():overlap+=sum(ids[k:k+len(q)]==q for k in range(max(0,len(ids)-len(q)+1)))
assert overlap==0
config=dict(experiment='H1036',training='First192 previously designated H998 discovery documents, all original BSF train. Recapture first16 tokens under current facade; historical data, no new corpus.',documents=docs,rank=128,arms=['task','natural_count_matched','natural_all'],natural_tokens=3072,count_matched_tokens=count,selection_seed=1036,other_ports='Native; residual-only representation comparison',evaluation='H1035 competent historical copying/spatial cohorts; diagnostic, not unopened validation',limits='Natural vs task differs in content, position distribution, centering and weighting. Count-matched arm controls sample count only; no attribution solely to diversity.',fit_holdout_rows=0)
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n')
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
# Reuse only pure suffix/metric definitions from the audited fixed factorial.
tree=ast.parse((R/'scripts/run_bilin18_projection_stream_factorial.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['suffix','KL','margin']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<audited-functions>','exec'),globals())
with torch.no_grad():
 captures=[]
 for start in range(0,192,8):
  ids=tokens[start:start+8].cuda();initial=F.rms_norm(model.transformer.wte(ids),(1152,)).float();raw,*_=runner.component._capture(model,initial,torch,F);captures.append(raw.cpu())
 X=torch.cat(captures).reshape(-1,1152).double().cuda();torch.save(dict(raw=X.cpu(),tokens=tokens,documents=docs),out/'training_captures.pt');fits={};gen=torch.Generator().manual_seed(1036);chosen=torch.randperm(len(X),generator=gen)[:count]
 pca=torch.load(A/'multistream_span_diagnosis/pca.pt',weights_only=False);fits['task']=(pca['center'][0].float().cuda(),pca['bases'][0].float().cuda())
 orth=0.
 for name,z in [('natural_count_matched',X[chosen.cuda()]),('natural_all',X)]:
  mu=z.mean(0,keepdim=True);c=z-mu;ev,b=torch.linalg.eigh(c.T@c/len(c));b=b[:,-128:].flip(1);orth=max(orth,float((b.T@b-torch.eye(128,device='cuda')).abs().max()));fits[name]=(mu.float(),b.float())
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
audit=dict(original_BSF_nontrain_documents_used=0,exact_eval_token_subsequence_overlap=overlap,prefix_sequences=192,suffix_sequences=len(records)*2,PCA_orthogonality_max=orth,task_baseline_summary_difference=cross,finite=all(np.isfinite(r['whole_KL']) and np.isfinite(r['paired_KL']) for r in records))
assert orth<1e-10 and cross<1e-4 and audit['finite']
torch.save({name:dict(center=mu.cpu(),basis=b.cpu()) for name,(mu,b) in fits.items()},out/'fits.pt');torch.save(samples,out/'audit_logits.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,energies=energies,audits=audit),indent=2)+'\n');print(json.dumps(audit,indent=2))

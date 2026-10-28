"""H1038 frozen basis same-family lexical transfer."""
from pathlib import Path
import sys,json,hashlib,ast,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_same_family_basis_transfer';out=A/'same_family_basis_transfer';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
import run_subject_number_sparse_graph_token_extraction_v1 as runner
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
train=torch.load(A/'multitask_boundary_contract/states.pt',weights_only=False)
held=torch.load(A/'multistream_sparse_baselines/development_states.pt',weights_only=False);pairs=json.loads((A/'multistream_sparse_baselines/development_panel.json').read_text());assert len(pairs)==19
ix=sorted({i for p in pairs for i in p['indices']});assert len(ix)==38
gates={(p['family'],p['split']) for p in pairs}
assert not {e['text'] for e in train['examples'].values()}&{e['text'] for e in held['examples'].values()}
files=[A/'paired_difference_basis_transfer/fits.pt',A/'multistream_sparse_baselines/development_states.pt',A/'multistream_sparse_baselines/development_panel.json'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
config=dict(experiment='H1038',rank=128,training='None; frozen H1037 activation and paired-difference bases.',arms=['task','paired_difference','mean_only_control'],source_hashes=hashes,other_ports='Native; paired edits transfer donor other ports exactly.',evaluation='All19 H1031 lexical-development pairs, five training families; historically opened. Capital has only3 equal-length pairs and is diagnostic.',center='Same frozen training activation center for both bases.',fit_rows=0,control='Mean-only residual decoder: paired residual change is zero; donor other ports still transferred. Added after initial two-arm run to disambiguate native assistance; not a tuned candidate.')
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n')
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
# Reuse only pure suffix/metric definitions from the audited fixed factorial.
tree=ast.parse((R/'scripts/run_bilin18_projection_stream_factorial.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['suffix','KL','margin']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<audited-functions>','exec'),globals())
with torch.no_grad():
 saved=torch.load(files[0],weights_only=False);fits={name:(r['center'].cuda(),r['basis'].cuda()) for name,r in saved.items()};fits['mean_only_control']=(fits['task'][0],torch.zeros_like(fits['task'][1]))
 states={i:tuple(x.cuda() for x in held['states'][i]) for i in ix};native={i:held['native'][i].cuda() for i in ix};records=[];samples=[];energies=[];replay=max(float((suffix(states[i])-native[i]).abs().max()) for i in ix);assert replay<1e-5
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
competence=[];delta_energy=[]
for f,sp in sorted(gates):
 pp=[p for p in pairs if (p['family'],p['split'])==(f,sp)];top=[];both=[]
 for p in pp:
  both.append(all(margin(native[i],held['examples'][i]['groups'])*(1 if end==0 else -1)>0 for end,i in enumerate(p['indices'])))
  top.extend(int(native[i].argmax()) in held['examples'][i]['groups'][end] for end,i in enumerate(p['indices']))
 competence.append(dict(family=f,split=sp,pairs=len(pp),both_endpoint_accuracy=float(np.mean(both)),top1=float(np.mean(top))))
for arm,(_,b) in fits.items():
 for p in pairs:
  i,d=p['indices'];delta=(states[d][0]-states[i][0]).reshape(-1,1152).double();delta_energy.append(dict(arm=arm,family=p['family'],pair=p['id'],retained=float((delta@b.double()).square().sum()/delta.square().sum())))
audit=dict(fit_rows=0,prefix_sequences=0,native_logit_replay_max=replay,source_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),suffix_sequences=len(records)*2+len(ix),train_eval_exact_prompt_overlap=0,finite=all(np.isfinite(r['whole_KL']) and np.isfinite(r['paired_KL']) for r in records))
assert audit['finite'] and audit['source_hashes_unchanged']
torch.save(samples,out/'audit_logits.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,competence=competence,records=records,energies=energies,difference_energy=delta_energy,audits=audit),indent=2)+'\n');print(json.dumps(audit,indent=2))

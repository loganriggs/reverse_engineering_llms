from pathlib import Path
import sys,json,hashlib
import numpy as np
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
sys.path.insert(0,str(repo/'basis_aligned/bilinear_quotient/ops'))
sys.path.insert(0,str(repo/'basis_aligned/polynomial_causal'))
sys.path.insert(0,str(repo))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
out=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition/prior_repo_bridge')
out.mkdir(exist_ok=True)
runner.OUT=out/'subject_graph_replay.json'
captured=[]
original=runner.component.evaluate
def capture(*args,**kwargs):
 result=original(*args,**kwargs)
 captured.append({k:result[k].detach().cpu().numpy() for k in ['atoms','prediction','target','base_margin']})
 return result
runner.component.evaluate=capture
import bilin18_observed_model_facade as facade
load_native=facade.load_bilin18
def load_local(**kwargs):
 return load_native(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),**kwargs)
facade.load_bilin18=load_local
runner.main()
summary={}
arrays={}
for name,batches in [('original',captured[:2]),('historical_fresh',captured[2:])]:
 atoms=np.concatenate([x['atoms'] for x in batches],axis=1)
 target=np.concatenate([x['target'] for x in batches])
 additive=atoms[[i for i,m in enumerate(runner.component.MASKS) if m.bit_count()==1]].sum(0)
 summary[name]={'rows':len(target),'additive':runner.metric(additive,target),'nine_edge':runner.metric(atoms.sum(0),target),'pair_correction_rms':float(np.sqrt(np.mean((atoms.sum(0)-additive)**2)))}
 arrays[name+'_atoms']=atoms;arrays[name+'_target']=target
np.savez(out/'subject_graph_effects.npz',**arrays)
summary['scope']='Replay of previously opened historical panels, not new OOD evidence. Targets require native intervention corners; no featurizer trained.'
(out/'subject_graph_baselines.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

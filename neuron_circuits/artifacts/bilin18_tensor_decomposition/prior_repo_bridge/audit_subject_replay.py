from pathlib import Path
import json,numpy as np
p=Path(__file__).parent
new=json.loads((p/'subject_graph_replay.json').read_text())
old=json.loads(Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/basis_aligned/polynomial_causal/SUBJECT_NUMBER_SPARSE_GRAPH_TOKEN_EXTRACTION_V1_RESULT.json').read_text())
a=np.load(p/'subject_graph_effects.npz');deltas=[]
for panel in ['original_authority','fresh_authority']:
 for cell,r in old[panel]['reports'].items():
  deltas.append(abs(r['relative_l2']-new[panel]['reports'][cell]['relative_l2']))
assert all(new['predictions'].values())
checks={}
for name in ['original','historical_fresh']:
 atoms=a[name+'_atoms'];target=a[name+'_target'];assert np.isfinite(atoms).all()
 checks[name]=float(np.sqrt(np.sum((np.sum(atoms,axis=0)-target)**2)/np.sum(target**2)))
assert max(deltas)<1e-4
result={'historical_cell_error_max_absolute_difference':max(deltas),'independent_nine_edge_relative_l2':checks,'original_binding_and_checkpoint_checks_passed':True,'historical_results_unchanged':True,'new_generalization_evidence':False}
(p/'subject_graph_replay_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

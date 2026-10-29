import json
from pathlib import Path
import numpy as np
import audit_readout_speed as audit
from quartic_cross_gram import product_factors,cross_gram

audit.ROOT=Path('results/readout_speed_confirmation');audit.SOURCE=audit.ROOT/'data';audit.CLOCK_TARGET=20.
audit.main()
generation=[]
for seed in range(249101,249106):
 data=np.load(audit.SOURCE/f'{seed}_data.npz');rows=np.r_[data['teacher_A'],data['teacher_B']];W=data['teacher_W']
 np.testing.assert_allclose(audit.literal(rows,data['x'])@W[:10].T+.05*data['noise'],data['y'],atol=1e-11,rtol=1e-10)
 np.testing.assert_allclose(audit.literal(rows,data['xa'])@W[10:].T+.05*data['adaptation_noise'],data['ya'],atol=1e-11,rtol=1e-10)
 F=product_factors(rows[:4],rows[4:],np.ones(4));G=cross_gram(F,F)
 np.testing.assert_allclose(np.einsum('tp,pq,tq->t',W,G,W),1.,atol=1e-10)
 generation.append(dict(seed=seed,source_tasks=10,heldout_tasks=4))
rows=json.loads((audit.ROOT/'results.json').read_text())
for r in rows:
 saved=np.load(audit.ROOT/r['checkpoint']);log=saved['logs'];failed=r['status']=='line_search_failed'
 trials=int(np.sum(log[:,4]+1))+(13 if failed else 0)
 assert r['gradient_calls']==r['steps']+int(failed)
 assert r['forward_calls']==trials+(r['steps'] if r['method']=='split' else 0)
 assert r['readout_solves']==1+(trials if r['method']=='reduced' else 0)
 assert r['flow_calls']==(r['steps'] if r['method']=='split' else 0)
 if r['status']=='clock_reached':np.testing.assert_allclose(r['hidden_clock'],20,atol=1e-10,rtol=0)
 rng=np.random.default_rng(r['init']);initial=rng.normal(size=(8,8));initial/=np.linalg.norm(initial,axis=1)[:,None]
 np.testing.assert_array_equal(initial,saved['initial_rows'])
report=json.loads((audit.ROOT/'audit.json').read_text());report['fresh_generation']=generation;report['resource_counters_checked']=True
(audit.ROOT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('Fresh generation and counters verified')

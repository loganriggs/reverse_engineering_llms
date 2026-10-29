"""Evidence inventory, not an assertion that the broad research goal is complete."""
import hashlib,json
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
checks=[]
for directory,field,count in [('e1_sweep','rows',240),('e1_matched','rows',240),('finite_readout_h4','fits',120),('scaling_grid','audited_fits',72),('depth5_alignment','fits',9),('instrumented_graph_recovery','explicit_DAG_checks',1280)]:
 p=Path('results')/directory/'audit.json';a=json.loads(p.read_text());assert a['status']=='passed';assert a[field]==count
 checked=[]
 if 'results_sha256' in a:
  result=p.parent/('results.csv' if directory.startswith('e1_') else 'results.json');assert sha(result)==a['results_sha256'];checked.append(str(result))
 checks.append(dict(area=directory,audit=str(p),audit_sha256=sha(p),verified_count=count,verified_result_hashes=checked))
svhn=list(Path('results/svhn_confirmation').rglob('audit.json'));assert len(svhn)==30
for p in svhn:
 a=json.loads(p.read_text());assert a['status']=='passed' and a['full_training_replayed']
 for rec in a['records']:
  assert sha(p.parent/rec['checkpoint'])==rec['sha256']
checks.append(dict(area='svhn_confirmation',verified_audits=30,verified_checkpoint_hashes=sum(len(json.loads(p.read_text())['records']) for p in svhn)))
# New optimizer panel has lists rather than a status flag.
p=Path('results/profiled_curvature/audit.json');a=json.loads(p.read_text());assert len(a)==20 and sum(r['audited_steps'] for r in a)==654
checks.append(dict(area='profiled_curvature',verified_fits=20,verified_steps=654,audit_sha256=sha(p)))
for directory,count_field,expected in [('larger_hidden_transfer','fits',192),('depth3_hidden_transfer','adaptation_fits',192),('depth3_measurement','new_readouts',48)]:
 p=Path('results')/directory/'audit.json';a=json.loads(p.read_text());assert a['status']=='passed' and a[count_field]==expected
 checks.append(dict(area=directory,audit=str(p),audit_sha256=sha(p),verified_count=expected,count_field=count_field))
result=dict(scope='Selected claim-to-evidence counts and provenance; not a fresh numerical replay of historical experiments or full completion proof',goal_complete=False,checks=checks,
 remaining=['Full proposed scaling envelope not executed','Quantitative robust feature recovery not established','Depth-three precision failures remain qualified','Larger fixed-width task coverage versus example budget not isolated','Generic physical circuit recovery remains unidentifiable without additional observations'])
out=Path('results/research_evidence');out.mkdir(exist_ok=True);(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Verified selected historical audit counts and SVHN checkpoint hashes; goal remains incomplete.')

"""M59 audit historical training calibration; no outcome-based fitting."""
from pathlib import Path
import json,hashlib,collections,numpy as np
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cp=V/'joint_shared_calibration.json';cal=json.loads(cp.read_text());audit=json.loads((V/'joint_shared_calibration_audit.json').read_text());assert audit['calibration_sha256']==sha(cp);assert cal['loss_source_sha256']==sha(P/'joint_shared_loss.py');targets=json.loads((V/'response_training_targets.json').read_text());protocol=json.loads((V/'joint_shared_comparison_protocol.json').read_text());assert protocol['target_source_sha256']==sha(V/'response_training_targets.json');groups=collections.defaultdict(list)
 for row in cal['rows']:
  families={targets['records'][i]['family'] for i in row['target_indices']};assert len(families)==1;assert len(row['generator_R'])==2;assert np.isclose(np.mean(row['generator_R']),row['R'],rtol=1e-12);groups[next(iter(families))].append(row)
 meanK=np.mean([r['K'] for r in cal['rows']]);meanR=np.mean([r['R'] for r in cal['rows']]);assert np.isclose(meanK,cal['pooled_K'],rtol=1e-12);assert np.isclose(meanR,cal['pooled_R'],rtol=1e-12);assert np.isclose(meanK/meanR,cal['response_weight'],rtol=1e-12);total=sum(r['R'] for r in cal['rows']);rows=[dict(family=f,scheduled_calibration_endpoints=len(rs),response_loss_share=sum(r['R'] for r in rs)/total,mean_R=np.mean([r['R'] for r in rs])) for f,rs in sorted(groups.items())];assert len(rows)==10 and all(r['scheduled_calibration_endpoints']==72 for r in rows)
 out=dict(status='passed',source_sha256=sha(Path(__file__)),calibration_sha256=sha(cp),loss_source_sha256=sha(P/'joint_shared_loss.py'),response_function_sha256=sha(P/'collect_response_targets.py'),training_source_sha256=sha(P/'train_joint_shared_comparison.py'),pooled_K=float(meanK),pooled_R=float(meanR),global_response_weight=cal['response_weight'],families=rows,scope='Equal family endpoint counts; mean raw native-weighted response squared errors, two generators per endpoint, one training-calibrated global K/R weight. No per-family native-energy normalization in loss. Initial scalar loss contributions are not gradients, training dynamics, or M57 evaluation response energy.')
 op=V/'response_calibration_balance.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print('PASSED',len(cal['rows']),'calibration endpoints')
if __name__=='__main__':main()

"""Bind all six new final states and twelve fixed controls to frozen evaluation."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ep=V/'encoded_change_evaluation_protocol.json';q=json.loads(ep.read_text());rp=V/'encoded_change_training_results.json';r=json.loads(rp.read_text());ap=V/'encoded_change_training_audit.json';a=json.loads(ap.read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);cp=V/'balanced_response_training_results.json';assert sha(cp)==q['controls_sha256'];old=json.loads(cp.read_text());models=[x for x in old['models'] if x['arm'] in ['intact_KL','KL_balanced_response']]+r['models'];assert len(models)==18;assert len({(x['kind'],x['seed'],x['arm']) for x in models})==18
 for x in models:assert sha(Path(x['path']))==x['sha256']
 for name,key in [('source_holdout_native_protocol.json','holdout_protocol_sha256'),('state_direction_factorial.json','factorial_reference_sha256'),('balanced_eval_natural.json','natural_reference_sha256'),('balanced_eval_train.json','training_reference_sha256')]:assert sha(V/name)==q[key]
 op=V/'encoded_change_evaluation_models.json';assert not op.exists();op.write_text(json.dumps(dict(status='all_final_states_bound',protocol_sha256=sha(ep),new_result_sha256=sha(rp),new_audit_sha256=sha(ap),controls_sha256=sha(cp),models=models,source_sha256=sha(Path(__file__)),scope='All six final encoded-change states plus exactly matched kind/seed KL-only and supplied-balanced controls. No model selection or performance-based exclusions.'),indent=2)+'\n');print('BOUND',18,'models')
if __name__=='__main__':main()

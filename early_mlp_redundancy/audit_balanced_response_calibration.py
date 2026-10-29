"""M60 independently aggregate training cell scales and scheduled calibration."""
from pathlib import Path
import json,hashlib,collections,numpy as np
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 pp=V/'balanced_response_training_protocol.json';p=json.loads(pp.read_text());cp=V/'balanced_response_calibration.json';c=json.loads(cp.read_text());assert c['protocol_sha256']==sha(pp);native=json.loads((V/'training_causal_coverage.json').read_text());old=json.loads((V/'training_coverage_featurizers.json').read_text());assert p['initial_results_sha256']==sha(V/'training_coverage_featurizers.json');a=json.loads((V/'training_coverage_featurizers_audit.json').read_text());assert a['status']=='passed' and a['result_sha256']==p['initial_results_sha256'];split=json.loads((V/'diverse_supervision_split.json').read_text());eligible=set(split['train_pair_ids']);eligible.update(x['pair_id'] for x in native['records'] if x['family']=='object_color');families={x['family'] for x in native['records']};assert len(families)==11
 for seed,s in p['schedules'].items():
  counts=collections.Counter(x['family'] for x in s['task_updates']);assert set(counts)==families and set(counts.values())=={6};assert len(s['natural_documents'])==66 and all(0<=x<32 for x in s['natural_documents']);assert all(t['pair_id'] in eligible and t['pair_id'] not in split['evaluation_pair_ids'] for t in s['task_updates'])
 energies={}
 for fam in families:
  for region in ['changed_input_positions','final_position']:
   rs=[x for x in native['records'] if x['family']==fam and x['region']==region and x['layer']==1 and x['status']=='evaluated'];energies[fam,region]=float(np.mean([x['response_energy'] for x in rs]))
 floor=.1*float(np.median([x for x in energies.values() if x>0]));assert np.isclose(floor,p['energy_floor'],rtol=1e-12)
 for s in p['scales']:assert np.isclose(s['native_mean_energy'],energies[s['family'],s['region']],rtol=1e-12) and np.isclose(s['denominator'],max(energies[s['family'],s['region']],floor),rtol=1e-12)
 calculated=[]
 for row in c['rows']:
  task=p['schedules'][str(row['seed'])]['task_updates'][row['task_update']];assert row['family']==task['family'] and row['key'].rsplit('|',1)[0]==task['pair_id'];b=next(b for b in old['baselines'] if b['key']==row['key'] and b['kind']==row['kind'] and b['seed']==row['seed'] and b['arm']=='intact_KL');rs=[x for x in old['rows'] if x['key']==row['key'] and x['kind']==row['kind'] and x['seed']==row['seed'] and x['arm']=='intact_KL' and x['region']!='identity' and x['status']=='evaluated'];raw=float(np.mean([x['response_error'] for x in rs]));balanced=float(np.mean([x['response_error']/max(energies[row['family'],x['region']],floor) for x in rs]));assert np.allclose([b['final_KL'],raw,balanced],[row['K'],row['R_raw'],row['R_balanced']],rtol=1e-12,atol=1e-12);calculated.append([b['final_KL'],raw,balanced])
 K,R,B=np.mean(calculated,axis=0);assert np.allclose([K,R,B,K/R,K/B],[c['pooled_K'],c['pooled_raw_R'],c['pooled_balanced_R'],c['raw_weight'],c['balanced_weight']],rtol=1e-12);assert len(calculated)==792;assert len({(r['kind'],r['seed'],r['task_update'],r['key']) for r in c['rows']})==792
 gradient=json.loads((V/'balanced_response_gradient_checks.json').read_text());assert gradient['status']=='passed' and gradient['protocol_sha256']==sha(pp) and gradient['calibration_sha256']==sha(cp);assert gradient['loss_source_sha256']==sha(P/'balanced_response_loss.py')
 out=dict(status='passed',source_sha256=sha(Path(__file__)),protocol_sha256=sha(pp),calibration_sha256=sha(cp),gradient_sha256=sha(V/'balanced_response_gradient_checks.json'),scheduled_rows=792,training_cells=22,floored_cells=sum(x['floored'] for x in p['scales']),raw_weight=float(K/R),balanced_weight=float(K/B),scope='Training-only schedules/scales/loss aggregation independently checked. No optimizer updates or natural-retention branch validation yet; task gradients passed separately.')
 op=V/'balanced_response_calibration_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()

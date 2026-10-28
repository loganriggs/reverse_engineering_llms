"""M23 one fixed first-moment ablation, preserving the M19 objective/data design."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 bp=P/'response_supervision_protocol.json';base=json.loads(bp.read_text());rp=P/'response_training_results.json';reference=json.loads(rp.read_text());assert reference['protocol_sha256']==sha(bp);assert json.loads((P/'response_training_audit.json').read_text())['result_sha256']==sha(rp)
 for x in base['initial_models']:assert sha(Path(x['path']))==x['sha256']
 p=dict(base,source_sha256=sha(Path(__file__)),parent_protocol_sha256=sha(bp),reference_training_sha256=sha(rp),optimizer='Fresh Adam beta1=0,beta2=.999,eps1e-8,lr.0003; same120updates and unit decoder columns; frozen normalization/nativeweights.',optimizer_betas=[0.,.999],comparison='Eightnewfits:fourM18expandedstarts x intactKL/plusresponse. Compare withinoptimizer objectives and withinobjective beta1=0 vs archivedM20beta1=.9. Identicalinitials,M19targets/calibratedweights,schedules,lr,steps,normalization,evaluationinventory and704M11-based nativevalidationdirections. No LR/beta/weight search. Removingfirstmoment also changesbiascorrection andrealstepmagnitudes; not a normmatched optimizer experiment.',stopping_rule='One fixed optimizer/objective comparison with alltrainingtargetretention, fullnativebenchmark,outputquadrants,naturalforgetting andfamilyeffects. No sentinel-specificselection. Preservefailures; do not automatically expand an optimizer grid. Bettertrainability alone doesnotprove semanticunits ornewtaskdiscovery.')
 op=P/'nomomentum_protocol.json'
 if op.exists():assert json.loads(op.read_text())==p
 else:op.write_text(json.dumps(p,indent=2)+'\n')
 for key in ['initial_models','schedules','arms','steps','learning_rate','target_design','response_loss','calibration','evaluation','gradient_checks']:assert p[key]==base[key]
 audit=dict(status='passed_before_training',protocol_sha256=sha(op),parent_protocol_sha256=sha(bp),reference_training_sha256=sha(rp),unchanged_fields=['initial_models','schedules','arms','steps','learning_rate','target_design','response_loss','calibration','evaluation','gradient_checks'],optimizer_betas=p['optimizer_betas'],scope='Protocol/hash equality to M19/M20; actualoptimizer configuration and finalmoment behavior must be checked by runner/audit.')
 (P/'nomomentum_protocol_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print('Frozen8fits; objective/data/evaluation unchanged; beta1=0')
if __name__=='__main__':main()

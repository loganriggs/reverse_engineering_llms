"""Paired time/work summaries, retaining individual seeds and order sensitivity."""
import json
from pathlib import Path
import numpy as np
from run_svhn_standard import digest,write

root=Path(__file__).parent
panels={}
for folder in ['bilinear_alternation','bilinear_alternation_long']:
    path=root/'results'/folder;audit=json.loads((path/'audit.json').read_text());assert audit['status']=='passed'
    rows=[json.loads(p.read_text()) for p in sorted(path.glob('*_0.*_*.json'))]
    assert len(rows)==108
    for row in rows:
        assert row['failure'] is None
        expected=next(r['artifact_sha256'] for r in audit['records'] if r['name']==row['name'])
        assert digest(path/f"{row['name']}.npz")==expected
    panels[folder]=rows
short={r['name']:r for r in panels['bilinear_alternation']};long=panels['bilinear_alternation_long']
for row in long:
    reference=short[row['name']]['curves'][-1]['mse']
    prefix=next(c for c in row['curves'] if c['time']==3)['mse']
    assert abs(reference-prefix)<1e-14
    for curve in short[row['name']]['curves']:
        assert curve == next(c for c in row['curves'] if c['sweep']==curve['sweep'])
summaries=[];paired=[]
for teacher in ['tied','untied']:
    for student in ['tied_raw','tied_corrected','untied_raw']:
        for lr in [.01,.003]:
            group=dict(teacher=teacher,student=student,lr=lr)
            for method in ['joint','forward','reverse']:
                rs=[r for r in long if all(r[k]==group[k] for k in ['teacher','student','lr']) and r['method']==method]
                group[method]=float(np.median([r['curves'][-1]['mse'] for r in rs]))
            summaries.append(group)
for row in long:
    if row['method']=='joint':continue
    baseline=next(r for r in long if r['method']=='joint' and all(r[k]==row[k] for k in ['teacher','student','lr','seed']))
    workpoint=min(row['curves'],key=lambda c:abs(c['time']-10))
    paired.append(dict(name=row['name'],method=row['method'],joint_final=baseline['curves'][-1]['mse'],
        alternating_equal_time=row['curves'][-1]['mse'],alternating_equal_work=workpoint['mse'],
        alternating_evaluations=3*workpoint['sweep'],joint_evaluations=baseline['full_gradient_evaluations']))
for method in ['forward','reverse']:
    selected=[r for r in paired if r['method']==method]
    assert len(selected)==36
    assert sum(r['alternating_equal_work']<r['joint_final'] for r in selected)==0
write(root/'results/bilinear_alternation_long/summary.json',dict(medians=summaries,paired=paired,deterministic_prefix_checks=108))
lines=['# Layerwise alternating gradients on depth-two bilinear networks','','216fits:108short and108long, with the entire short trajectory prefix reproduced in the long panel. Networks have input dimension3, hidden width3, top width3; tied39parameters versus untied57parameters. Bias-free and unnormalized. Teachers are tied or untied, with3paired teacher/initialization seeds per family. A/B directions are independently initialized; untied student occurrences copy the tied initial hidden parameters, giving identical initial functions without the previous equal-factor product symmetry.','',
    'Teacher and initial student Gaussian RMS are normalized to1 with frozen scales. Exact Gaussian population squared error removes sampling noise. This is an optimization study, not a sample-efficiency study. A tied student on an untied teacher is not automatically a certified unrepresentable case.','',
    'Blocks: all bottom A/B occurrences, top C/D, then readout w. Forward/reverse alternate these blocks; joint updates all simultaneously. Corrected tied updates divide bottom gradients by2. No momentum, clipping, decay, line search or RMSNorm. Rates .01/.003 and per-block times3/30 were applied to the whole panel.','',
    '| Teacher | Student / metric | Step | Joint MSE | Forward MSE | Reverse MSE |','|---|---|---:|---:|---:|---:|']
for r in summaries:
    lines.append(f"| {r['teacher']} | {r['student']} | {r['lr']} | {r['joint']:.7f} | {r['forward']:.7f} | {r['reverse']:.7f} |")
lines+=['','Table entries are median exact population MSE over3paired teacher/initialization seeds at per-block time30. Medians do not imply robust method rankings. For untied teacher seed211002 with raw tied student and step.01, forward/joint/reverse finish at.0441/.1682/.5616. At.003 they finish at.0900/.1858/.3540. Block order can strongly affect a difficult trajectory; its effect is not a universally beneficial acceleration.','',
    'Work control: this implementation computes one full gradient per joint sweep and three per alternating sweep. Comparing joint time30 against alternating time10 matches3000versus3000evaluations at step.01 and10000versus9999at.003. Forward beats joint in0/36pairs; reverse beats joint in0/36. This is a full-gradient-evaluation budget, not measured hardware FLOPs or an optimal specialized block-gradient implementation. Equal per-block time comparisons give alternation roughly3times the gradient work.','',
    'Independent validation: all216final losses replay through monomial expansion and exact Gaussian moments. Initial/final analytic gradients agree with autograd; three-sweep block updates agree with an independent autograd implementation. All108long prefixes exactly reproduce short endpoints. No nonfinite fits. Error magnitudes, checkpoints, source snapshots and all individual outcomes are saved.','',
    'Theory extension: multiplicity-corrected tied updates and occurrence-projected updates remain equivalent under any common block schedule that selects all occurrences of each shared parameter. Nine20-sweep fixtures agree within6.67e-16. Rowwise reciprocal-scaling gradient identities and finite-step norm-balance drift checks also pass. See SHARING_COMPLEXITY_AND_FLOW.md and results/alternating_projection/checks.json.','',
    'Conclusion: naive layerwise alternation changes some trajectories, but has not earned its extra gradient cost in this panel. This is neither a global-convergence result nor a rejection of larger dwell times, adaptive block methods, specialized implementations, or damped curvature-aware methods. Next prioritize an optimizer with a concrete conditioning rationale and independent work controls, alongside the already planned image data/capacity replication. No circuit-recovery claim follows from these function errors.','']
# Counts derived from the implementation rather than relying on textual guesses.
from experiment import count
assert count(3,3,3,True)==39 and count(3,3,3,False)==57
(root/'BILINEAR_ALTERNATION_RESULTS.md').write_text('\n'.join(lines))
print('\n'.join(lines))

"""Report all gauge controls, retaining failures and numerical stopping ties."""
import itertools,json,statistics
from pathlib import Path
import numpy as np
from run_bilinear_gauss_newton import jacobian,unpack
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/bilinear_gauge_balance'
audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed' and len(audit['records'])==54
rows=[];geometry=[]
z,w1=np.polynomial.hermite_e.hermegauss(5);ids=np.array(list(itertools.product(range(5),repeat=3)));x,w=z[ids],np.prod(w1[ids]/np.sqrt(2*np.pi),axis=1)
for a in audit['records']:
 p=out/(a['name']+'.json');assert digest(p)==a['result_sha256'] and digest(p.with_suffix('.npz'))==a['artifact_sha256']
 r=json.loads(p.read_text());rows.append(r);states=np.load(p.with_suffix('.npz'))
 if r['method']=='original':
  old=root/'results/bilinear_gauss_newton'/f"{r['teacher']}_{r['seed']}_{r['student']}_0.001.npz"
  np.testing.assert_array_equal(np.load(old)['states'],states['states'])
 if r['method']=='initial':
  d=np.load(root/r['input_file']);template={k[8:]:d[k] for k in d.files if k.startswith('initial_')}
  ratios=[]
  for key in ['rawstates','states']:
   J=np.sqrt(w)[:,None]*jacobian(unpack(states[key][0],template),x,float(d['student_scale']))
   sv=np.linalg.svd(J,compute_uv=False);ratios.append(float(sv[0]/sv[14]))
  geometry.append(dict(name=r['name'],original_nonzero_condition=ratios[0],balanced_nonzero_condition=ratios[1]))
lines=['# Function-preserving factor balancing','','54 fixed-damping Gauss–Newton fits: 18 archived teacher/student/metric fixtures times original, initial-only balancing, and balancing initially plus after every accepted step. Same initial function, frozen normalization, damping .001, Armijo schedule, 100-Jacobian cap and MSE<1e-12 stopping criterion. No restarts or damping tuning. Balance paired rows A/B, AR/BR when present, and C/D by reciprocal positive scaling. Zero-norm pairs are left unchanged. The readout is unchanged.','', '| Student | Balance schedule | Median population MSE | Below 1e-12 | Median Jacobian calls |','|---|---|---:|---:|---:|']
for student in ['tied_raw','tied_corrected','untied_raw']:
 for method in ['original','initial','every']:
  rs=[r for r in rows if r['student']==student and r['method']==method]
  lines.append(f"| {student} | {method} | {statistics.median(r['curves'][-1]['mse'] for r in rs):.6g} | {sum(r['curves'][-1]['mse']<1e-12 for r in rs)}/6 | {statistics.median(r['jacobian_evaluations'] for r in rs):g} |")
comparisons=[]
for method in ['initial','every']:
 wins=ties=losses=0
 for r in rows:
  if r['method']!=method:continue
  f=next(f for f in rows if f['method']=='original' and all(f[k]==r[k] for k in ['teacher','seed','student']))
  a,b=r['curves'][-1]['mse'],f['curves'][-1]['mse'];tie=a<1e-12 and b<1e-12
  ties+=tie;wins+=not tie and a<b;losses+=not tie and a>=b
 comparisons.append(dict(method=method,wins=wins,ties=ties,losses=losses))
 lines.append(f'\n{method}: {wins} lower-error outcomes, {ties} tolerance ties, {losses} higher-error outcomes versus original coordinates across 18 paired fixtures.')
improved=sum(g['balanced_nonzero_condition']<g['original_nonzero_condition'] for g in geometry)
lines+=['',f'At initialization, balancing improves the condition number of the 15 nonzero weighted-Jacobian singular directions in {improved}/18 fixtures. These include repeated initial parameterizations across teacher families and metrics, so they are not 18 independent draws. Ratios and individual outcomes are saved in summary.json. This diagnostic alone does not explain final performance.','',
'One failure is retained: untied teacher seed211003, raw tied student, every-step balancing exhausted all 12 line-search candidates at iteration98, MSE1.9970e-5. The independent audit reproduces the descent direction and rejection of every candidate. No retry was performed. Other fits remain finite; not every successful run reaches stopping tolerance.','',
'## What balancing does mathematically','',
'For nonzero paired vectors a,b, set c=sqrt(||b||/||a||). Then (ca)(b/c)^T=ab^T and both new norms equal sqrt(||a||||b||). This minimizes c²||a||²+c⁻²||b||² along the positive reciprocal-scaling orbit, by AM–GM or differentiation. It fixes this particular scale freedom, not all network symmetries. Equal parameter norm does not guarantee an optimally conditioned output Jacobian.','',
'Keeping the same numerical damping matrix after rescaling deliberately changes the optimizer metric. If theta=T phi is a fixed invertible change of coordinates, transporting damping to TᵀMT would preserve the corresponding linearized update; using I or the same multiplicity matrix in the new coordinates does not. Thus these are algorithmic interventions on geometry, not an assertion that Gauss–Newton is coordinate invariant. Rebalancing after each step is state dependent.','',
'## Verification and decision','',
'All 18 original trajectories reproduce archived GN states exactly. Every rebalanced state has independently matching polynomial coefficients and paired outer products before/after balancing; applicable row norms match. All trajectory losses replay through exact Gaussian coefficient moments. Audits verify raw pre-balance step equations, Armijo descent, sampled autograd Jacobians and the failed final search. Both raw and balanced states are retained.','',
'Initial-only balancing improves the tied-model median errors, but neither schedule universally wins. This supports sensitivity to function-preserving coordinates and does not prove that reciprocal row imbalance accounts for all difficult trajectories. These are tiny population-regression tasks, not sample-efficiency or circuit-recovery results. Close the bounded geometry controls; next prioritize multiple-readout/intervention supervision and held-out transfer to address hidden-computation recovery, rather than expanding optimizer rescue sweeps.','']
(root/'BILINEAR_GAUGE_BALANCE_RESULTS.md').write_text('\n'.join(lines))
write(out/'summary.json',dict(status='complete',comparisons=comparisons,geometry=geometry,source_sha256=digest(Path(__file__))))
print('\n'.join(lines))

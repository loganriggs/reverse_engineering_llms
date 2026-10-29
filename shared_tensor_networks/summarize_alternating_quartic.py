"""Summaries and independently integrated simultaneous-flow reference."""
import itertools,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from run_alternating_quartic import gradient
from run_svhn_standard import digest,write

root=Path(__file__).parent
z,w1=np.polynomial.hermite_e.hermegauss(5);ids=np.array(list(itertools.product(range(5),repeat=3)))
x,w=z[ids],np.prod(w1[ids]/np.sqrt(2*np.pi),axis=1)
references={};reference_rows=[]
for seed in [170001,170002,170003]:
    d=np.load(root/f'results/fixed_degree_reuse/t4_s{seed}_symmetric_g4.npz')
    target=(x@d['teacher'].T).prod(1)/float(d['teacher_scale']);scale=float(d['student_scale'])
    def rhs(t,a):return -gradient(a.reshape(4,3),x,w,target,scale).ravel()
    solutions=[solve_ivp(rhs,[0,30],d['initial'].ravel(),method='DOP853',rtol=tol,atol=tol*.01) for tol in [1e-10,1e-12]]
    assert all(s.success for s in solutions)
    error=float(np.max(abs(solutions[0].y[:,-1]-solutions[1].y[:,-1])));assert error<1e-8
    a=solutions[1].y[:,-1].reshape(4,3);references[seed]=a
    reference_rows.append(dict(seed=seed,tolerance_comparison_max_error=error,
        mse=float(w@((x@a.T).prod(1)/scale-target)**2),factor_spread=float(np.linalg.norm(a-a.mean(0)))))
rows=[];individual=[]
for family in ['alternating_quartic','alternating_quartic_flow']:
    folder=root/'results'/family
    audit=json.loads((folder/'audit.json').read_text());assert audit['status']=='passed'
    files=sorted(folder.glob('s*_lr*.json'))
    for file in files:
        r=json.loads(file.read_text());d=np.load(file.with_suffix('.npz'))
        expected=next(item['artifact_sha256'] for item in audit['records'] if item['name']==r['name'])
        assert digest(file.with_suffix('.npz'))==expected
        individual.append(dict(family=family,seed=r['seed'],lr=r['lr'],mode=r['mode'],
            mse=r['curves'][-1]['population_mse'],spread=r['curves'][-1]['factor_spread'],
            first_sweep_spread=r['curves'][1]['factor_spread'],
            distance_to_flow=float(np.linalg.norm(d['student']-references[r['seed']])),
            recovered=r['recovery']['strict_recovery']))
    modes=['simultaneous','forward','reverse','random'] if family=='alternating_quartic' else ['forward','reverse']
    for lr in [.03,.01,.003]:
        for mode in modes:
            selected=[r for r in individual if r['family']==family and r['lr']==lr and r['mode']==mode]
            rows.append(dict(family=family,lr=lr,mode=mode,
                median_mse=float(np.median([r['mse'] for r in selected])),
                median_spread=float(np.median([r['spread'] for r in selected])),
                median_distance_to_flow=float(np.median([r['distance_to_flow'] for r in selected])),
                recovered=sum(r['recovered'] for r in selected)))
write(root/'results/alternating_quartic/summary.json',dict(rows=rows,individual=individual,flow_references=reference_rows))
lines=['# Alternating updates and exact block gradient flow','','54fits:36gradient-step fits and18exact frozen-block-flow fits. Three archived generic quartic teachers, exactly symmetric four-factor initializations, rates/dwell times .03/.01/.003, final per-block time30. All objectives are exact Gaussian population MSE via polynomial-exact quadrature. No new sample noise, no restarts, no validation tuning.','',
    'Each sweep updates every factor once; the simultaneous gradient step evaluates predictions once, alternating gradients four times, and exact block flow solves four3x3 affine ODEs. These are matched per-block clocks, not matched wall time or FLOPs. For a switched-flow physical clock, a four-block cycle lasts4eta; rescale time by4 when comparing with simultaneous flow.','',
    '| Method | Step/dwell | Median MSE | Median factor spread | Distance to simultaneous flow | Recovery |',
    '|---|---:|---:|---:|---:|---:|']
for r in rows:
    label=('gradient '+r['mode']) if r['family']=='alternating_quartic' else ('exact block flow '+r['mode'])
    lines.append(f"| {label} | {r['lr']} | {r['median_mse']:.8f} | {r['median_spread']:.6g} | {r['median_distance_to_flow']:.6g} | {r['recovered']}/3 |")
lines+=['','Cyclic updates break exact factor equality, but the benefit is small at this budget and no fit passes the existing strict factor-recovery criterion. Forward/reverse sweeps agree in prediction error here because identical initialization and a permutation-symmetric product objective make them relabelings; this is not an order-independence theorem for arbitrary architectures. Random order gives weaker persistent splitting in this panel.','',
    'Exact block flow shows the same qualitative effect: symmetry breaking is not solely an Euler discretization artifact. Switching whole blocks for finite dwell time itself changes the trajectory. The deviations shrink as dwell time shrinks, consistent with the simultaneous-flow limit after clock rescaling. Simultaneous flow was integrated independently with DOP853 at two tolerances; endpoint discrepancy is below1e-8.','',
    'Validation: all54final errors independently replay through polynomial coefficients and analytic Gaussian moments. Initial/final analytic gradients agree with autograd. Five-sweep discrete trajectories agree with independent autograd updates; exact-flow trajectories agree with augmented-matrix exponentials built from autograd block Hessians. Maximum discrepancies are in the saved audit records. No nonfinite fits occurred. These tests support correctness, not general convergence.','',
    'Conclusion: finite block switching can release an exact symmetry constraint, but does not by itself deliver useful factor recovery in these runs. This bounded test does not justify claiming alternating dynamics are superior. Next distinguish practical layerwise conditioning effects from this small symmetry mechanism, using independent initializations and the actual bilinear architecture; retain step-size and work controls.','']
(root/'ALTERNATING_DYNAMICS_RESULTS.md').write_text('\n'.join(lines))
print('\n'.join(lines))

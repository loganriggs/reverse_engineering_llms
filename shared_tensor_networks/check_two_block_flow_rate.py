"""Numerical checks of the proved two-scalar-block flow comparison."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
rng=np.random.default_rng(268101);records=[];matrix_error=trace_error=ode_error=0.
for fixture in range(24):
 a,b=10**rng.uniform(-2,2,2)
 for q in [0.,.01,.25,.81,.9801]:
  c=(-1 if fixture%2 else 1)*np.sqrt(q*a*b);H=np.array([[a,c],[c,b]])
  for scale in [.01,.1,1.,10.]:
   tau=scale/(a+b);u,v=np.exp(-a*tau),np.exp(-b*tau)
   A=np.array([[u,-c/a*(-np.expm1(-a*tau))],[0.,1.]])
   B=np.array([[1.,0.],[-c/b*(-np.expm1(-b*tau)),v]])
   C=B@A;independent=expm(-tau*np.diag([0.,1.])@H)@expm(-tau*np.diag([1.,0.])@H)
   np.testing.assert_allclose(C,independent,rtol=1e-11,atol=1e-12);matrix_error=max(matrix_error,float(np.max(abs(C-independent))))
   trace=u+v+q*(1-u)*(1-v);trace_error=max(trace_error,abs(trace-np.trace(C)))
   eig=np.linalg.eigvals(C);assert np.max(abs(eig.imag))<1e-12;rho=float(max(eig.real));joint=float(np.exp(-min(np.linalg.eigvalsh(H))*tau))
   assert rho>=joint-2e-12 and 0<rho<=1+1e-12
   if q==0:np.testing.assert_allclose(C,expm(-tau*H),rtol=1e-12,atol=1e-12)
   if fixture<4 and q==.25 and scale==1.:
    z=rng.normal(size=2);left=solve_ivp(lambda t,z:-(np.diag([1.,0.])@H)@z,[0,tau],z,method='DOP853',rtol=1e-12,atol=1e-13).y[:,-1]
    endpoint=solve_ivp(lambda t,z:-(np.diag([0.,1.])@H)@z,[0,tau],left,method='DOP853',rtol=1e-12,atol=1e-13).y[:,-1]
    np.testing.assert_allclose(endpoint,C@z,rtol=1e-10,atol=1e-11);ode_error=max(ode_error,float(np.max(abs(endpoint-C@z))))
   records.append(dict(a=float(a),b=float(b),c=float(c),tau=float(tau),alternating_radius=rho,joint_radius=joint))
a=b=1.;c=.5;tau=1.;u=np.exp(-1);A=np.array([[u,-.5*(1-u)],[0,1.]]);B=np.array([[1.,0],[-.5*(1-u),u]])
example_eigenvalues=np.linalg.eigvals(B@A);assert max(abs(example_eigenvalues.imag))<1e-12
example=dict(a=a,b=b,c=c,tau=tau,alternating_radius=float(max(example_eigenvalues.real)),joint_radius=float(np.exp(-.5)))
out=Path('results/two_block_flow_rate');out.mkdir(exist_ok=False)
result=dict(scope='Two-scalar-block strictly convex quadratic only; not a network-training experiment',fixtures=len(records),matrix_exponential_max_error=matrix_error,trace_max_error=float(trace_error),ode_max_error=ode_error,example=example,records=records,sources={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['check_two_block_flow_rate.py','TWO_BLOCK_FLOW_RATE.md']})
(out/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','sources']}),flush=True)

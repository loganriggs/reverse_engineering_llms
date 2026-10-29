"""Exact algebra behind the two-feature scalar-identification argument."""
from pathlib import Path
import sympy as s
from run_svhn_standard import digest,write
u,v=s.symbols('u v');x=s.symbols('x0:4');h1=x[0]*x[1];h2=x[2]*x[3]
Q=s.hessian(u*h1+v*h2,x)/2
assert s.factor(Q.det())==u**2*v**2/16
assert Q.subs({u:1,v:0}).rank()==Q.subs({u:0,v:1}).rank()==2
assert Q.subs({u:1,v:1}).rank()==4
# Nonorthogonal independent factors preserve rank and the two rank-two lines.
R=s.Matrix([[1,1,0,0],[0,1,1,0],[0,0,1,1],[1,0,0,2]])
assert R.det()!=0
assert s.factor((R.T*Q*R).det())==R.det()**2*u**2*v**2/16
# A nondegenerate scalar readout splits into two distinct full-rank quadratics.
f=(h1+h2)*(h1+2*h2)
assert s.expand(f-(h1**2+3*h1*h2+2*h2**2))==0
assert (s.hessian(h1+h2,x)/2).rank()==(s.hessian(h1+2*h2,x)/2).rank()==4
out=Path(__file__).parent/'results/readout_recovery'
write(out/'two_feature_checks.json',dict(status='passed',pencil_determinant=str(s.factor(Q.det())),coordinate_determinant=str(R.det()),source_sha256=digest(Path(__file__))))
print('Exact pencil determinant, ranks, coordinate transform and scalar factorization passed.')

"""Exact depth-three functional ambiguity with distinct internal interventions."""
import json,hashlib
from pathlib import Path
import sympy as s
z1,z2,x1,x2,x3,x4=s.symbols('z1 z2 x1 x2 x3 x4')
u=z1*z1-z2*z2;v=2*z1*z2;tasks=s.Matrix([u*u,u*v,v*v]);records=[]
for r in [s.Rational(1),s.Rational(2),s.Rational(3),s.Rational(1,2)]:
 t=(r-1/r)/2;g=(z1+r*z2)*(z1-z2/r);assert s.expand(g-u-t*v)==0
 features=s.Matrix([g*g,g*v,v*v]);readout=s.Matrix([[1,-2*t,t*t],[0,1,-t],[0,0,1]])
 error=[s.expand(q) for q in readout*features-tasks];assert error==[0,0,0]
 expanded=[s.Poly(q.subs({z1:x1*x2,z2:x3*x4}),x1,x2,x3,x4) for q in tasks]
 assert all(q.total_degree()==8 for q in expanded)
 coefficients=s.Matrix([[s.Poly(q,z1,z2).coeff_monomial(z1**(4-j)*z2**j) for j in range(5)] for q in tasks]);assert coefficients.rank()==3
 # Same numbered operation: zero the first second-layer feature, keep the second.
 original_ablation=s.Matrix([0,0,v*v]);changed_ablation=readout*s.Matrix([0,0,v*v])
 intervention_delta=[s.expand(q) for q in changed_ablation-original_ablation]
 if t!=0:assert intervention_delta[0]!=0 and intervention_delta[1]!=0
 records.append(dict(r=str(r),t=str(t),factorization=str(g),readout=[[str(k) for k in row] for row in readout.tolist()],
                     identical_task_coefficients=True,task_span_rank=3,degree=8,ablation_difference=[str(k) for k in intervention_delta]))
# All 17 parameters of a full-product depth-three readout, first layer fixed.
a=s.symbols('a0:4');b=s.symbols('b0:4');w=s.symbols('w0:9');parameters=list(a+b+w)
k=[(a[2*i]*z1+a[2*i+1]*z2)*(b[2*i]*z1+b[2*i+1]*z2) for i in range(2)]
p=s.Matrix([k[0]**2,k[0]*k[1],k[1]**2]);outputs=s.Matrix(3,3,w)*p
coef=s.Matrix([s.Poly(q,z1,z2).coeff_monomial(z1**(4-j)*z2**j) for q in outputs for j in range(5)])
point=[1,-1,2,0,1,1,0,1]+list(s.eye(3))
J=coef.jacobian(parameters).subs(dict(zip(parameters,point)));rank=J.rank();assert rank==11
# Independent evaluated-output differentiation: each individual parameter has degree <=2.
def evaluated(values):
 out=[]
 for task in range(3):
  for z in range(5):
   hidden=[(values[2*i]+values[2*i+1]*z)*(values[4+2*i]+values[5+2*i]*z) for i in range(2)]
   products=[hidden[0]**2,hidden[0]*hidden[1],hidden[1]**2]
   out.append(sum(values[8+3*task+j]*products[j] for j in range(3)))
 return s.Matrix(out)
V=s.diag(*[s.Matrix([[z**j for j in range(5)] for z in range(5)])]*3)
for step in [1,2]:
 for col in range(17):
  plus=point.copy();minus=point.copy();plus[col]+=step;minus[col]-=step
  assert (evaluated(plus)-evaluated(minus))/(2*step)==(V*J)[:,col]
cols=list(J.rref()[1]);rows=list(J[:,cols].T.rref()[1]);minor=J.extract(rows,cols).det();assert minor!=0
# Explicit usual four scale gauges, separate from the two mixing degrees.
gauges=[]
for i in range(2):
 g=s.zeros(17,1)
 for j in range(2):g[2*i+j]=point[2*i+j];g[4+2*i+j]=-point[4+2*i+j]
 gauges.append(g)
 g=s.zeros(17,1)
 for j in range(2):g[2*i+j]=point[2*i+j]
 for task in range(3):
  for col,(left,right) in enumerate([(0,0),(0,1),(1,1)]):g[8+3*task+col]=-point[8+3*task+col]*(int(left==i)+int(right==i))
 gauges.append(g)
G=s.Matrix.hstack(*gauges);assert G.rank()==4 and J*G==s.zeros(15,4)
root=Path('results/depth3_mixing');root.mkdir(parents=True,exist_ok=True)
result=dict(exact_examples=records,parameters=17,coefficient_outputs=15,jacobian_rank=rank,local_fiber_dimension=17-rank,
 scaling_gauge_rank=4,additional_local_dimensions=17-rank-4,minor_rows=rows,minor_columns=cols,minor_determinant=str(minor),
 jacobian=[[str(k) for k in row] for row in J.tolist()],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(root/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
print('Exact examples',len(records),'Jacobian rank',rank,'fiber dimension',17-rank,'beyond scaling',17-rank-4,'minor',minor)

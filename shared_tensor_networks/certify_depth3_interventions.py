"""Exact local ranks for observing outputs after zeroing one internal feature."""
import hashlib,itertools,json
from pathlib import Path
import sympy as s
z1,z2=s.symbols('z1 z2');a=s.symbols('a0:4');b=s.symbols('b0:4');w=s.symbols('w0:9');params=list(a+b+w)
h=[(a[2*i]*z1+a[2*i+1]*z2)*(b[2*i]*z1+b[2*i+1]*z2) for i in range(2)];W=s.Matrix(3,3,w)
original=W*s.Matrix([h[0]**2,h[0]*h[1],h[1]**2]);ablated=W*s.Matrix([0,0,h[1]**2])
def coefficients(outputs):return s.Matrix([s.Poly(q,z1,z2).coeff_monomial(z1**(4-j)*z2**j) for q in outputs for j in range(5)])
point=[1,-1,2,0,1,1,0,1]+list(s.eye(3));subs=dict(zip(params,point));records=[]
for count in range(4):
 for subset in itertools.combinations(range(3),count):
  expr=coefficients(list(original)+[ablated[t] for t in subset]);J=expr.jacobian(params).subs(subs)
  rank=J.rank();cols=list(J.rref()[1]);rows=list(J[:,cols].T.rref()[1]);det=J.extract(rows,cols).det();assert det!=0
  # Independent integer evaluated-output forward differences.
  def forward(v):
   result=[]
   for task,is_ablation in [(t,False) for t in range(3)]+[(t,True) for t in subset]:
    for z in range(5):
     k=[(v[2*i]+v[2*i+1]*z)*(v[4+2*i]+v[5+2*i]*z) for i in range(2)]
     if is_ablation:k[0]=0
     products=[k[0]**2,k[0]*k[1],k[1]**2]
     result.append(sum(v[8+3*task+j]*products[j] for j in range(3)))
   return s.Matrix(result)
  V=s.diag(*[s.Matrix([[z**j for j in range(5)] for z in range(5)])]*(3+count))
  for step in [1,2]:
   for col in range(17):
    plus=point.copy();minus=point.copy();plus[col]+=step;minus[col]-=step
    assert (forward(plus)-forward(minus))/(2*step)==(V*J)[:,col]
  assert rank<=13
  records.append(dict(observed_ablated_task_indices=list(subset),rank=rank,fiber_dimension=17-rank,beyond_four_scales=13-rank,minor_rows=rows,minor_columns=cols,minor=str(det)))
assert records[0]['rank']==11 and records[-1]['rank']==13
tau,t0=s.symbols('tau t0',positive=True)
quartic=t0/s.sqrt(1+4*t0*t0*tau);quadratic=t0*s.exp(-tau)
assert s.simplify(s.diff(quartic,tau)+2*quartic**3)==0
assert s.simplify(s.diff(quadratic,tau)+quadratic)==0
assert quartic.subs(tau,0)==t0 and quadratic.subs(tau,0)==t0
root=Path('results/depth3_interventions');root.mkdir(parents=True,exist_ok=True)
(root/'certificate.json').write_text(json.dumps(dict(scalar_flow_identities=True,records=records,point=[str(v) for v in point],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
print([(r['observed_ablated_task_indices'],r['rank']) for r in records])

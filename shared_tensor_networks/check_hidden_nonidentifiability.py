"""Exact equal-width bilinear networks with equal outputs and different hidden spans."""
from pathlib import Path
import sympy as sp
import torch
from experiment import Network
from run_svhn_standard import write,digest

x,y=sp.symbols('x y',real=True)
h=sp.Matrix([x*x,y*y]);g=sp.Matrix([x*x-y*y,2*x*y])
f=(h[0]+h[1])**2;other=g[0]**2+g[1]**2
assert sp.expand(f-other)==0
teacher_coeff=sp.Matrix([[1,0],[0,0],[0,1]])
student_coeff=sp.Matrix([[1,0],[0,2],[-1,0]])
assert teacher_coeff.rank()==student_coeff.rank()==2
assert teacher_coeff.row_join(student_coeff).rank()==3
teacher_intervention=y**4;student_intervention=4*x*x*y*y
assert sp.expand(teacher_intervention-student_intervention)!=0
# Realize the symbolic identity in the same Network class used by the experiments.
torch.set_num_threads(1)
teacher=Network(2,2,2,True,0);student=Network(2,2,2,True,0)
with torch.no_grad():
    teacher.A.copy_(torch.eye(2));teacher.B.copy_(torch.eye(2));teacher.C.fill_(1);teacher.D.fill_(1);teacher.w.fill_(.5)
    student.A.copy_(torch.tensor([[1.,1.],[2.,0.]]));student.B.copy_(torch.tensor([[1.,-1.],[0.,1.]]));student.C.copy_(torch.eye(2));student.D.copy_(torch.eye(2));student.w.fill_(1)
    inputs=torch.randn(100,2,generator=torch.Generator().manual_seed(214101))
    torch.testing.assert_close(teacher(inputs),student(inputs),atol=1e-12,rtol=1e-12)
root=Path(__file__).parent;out=root/'results/hidden_nonidentifiability';out.mkdir(exist_ok=True)
write(out/'checks.json',dict(status='proved_and_checked',identity=str(sp.expand(f)),teacher_hidden=[str(v) for v in h],alternative_hidden=[str(v) for v in g],
    teacher_hidden_rank=2,alternative_hidden_rank=2,joint_hidden_rank=3,
    teacher_first_feature_ablation=str(teacher_intervention),alternative_first_feature_ablation=str(student_intervention),
    parameters_each=sum(p.numel() for p in teacher.parameters()),source_sha256=digest(Path(__file__)),dependency_sha256=digest(root/'experiment.py'),
    scope='exact illustrative radial target; not a generic identifiability theorem or a canonical matching of feature ablations'))
print('Exact identity; distinct hidden spans; equal18parameter architectures')

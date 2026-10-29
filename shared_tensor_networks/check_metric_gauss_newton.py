"""Occurrence-metric damping and coordinate covariance of tied Gauss-Newton steps."""
from pathlib import Path
import torch
from experiment import Network
from run_svhn_standard import write,digest

torch.set_num_threads(1)
rows=[]
for seed in [213001,213002,213003]:
    tied=Network(3,3,3,True,seed);free=Network(3,3,3,False,seed)
    with torch.no_grad():free.AR.copy_(tied.A);free.BR.copy_(tied.B)
    names=list(dict(tied.named_parameters()));offsets={};offset=0
    for name,p in tied.named_parameters():offsets[name]=(offset,p.numel());offset+=p.numel()
    chunks=[]
    for name,p in free.named_parameters():
        target={'AR':'A','BR':'B'}.get(name,name);start,n=offsets[target]
        block=torch.zeros(n,offset);block[:,start:start+n]=torch.eye(n);chunks.append(block)
    S=torch.cat(chunks);M=S.T@S;Q=S@torch.diag(M.diag().rsqrt())
    rng=torch.Generator().manual_seed(seed+1);x=torch.randn(11,3,generator=rng);r=torch.randn(11,generator=rng)
    y=free(x);J=torch.stack([torch.cat([g.flatten() for g in torch.autograd.grad(y[i],tuple(free.parameters()),retain_graph=True)]) for i in range(len(y))])
    T=torch.diag(torch.exp(torch.linspace(-1,1,offset)));A=J@S
    for damping in [.001,.1,1.]:
        step=torch.linalg.solve(A.T@A+damping*M,-A.T@r)
        tangent_step=Q@torch.linalg.solve((J@Q).T@(J@Q)+damping*torch.eye(offset),-(J@Q).T@r)
        error=float((S@step-tangent_step).norm()/tangent_step.norm());assert error<1e-9
        changed=A@T;changed_metric=T.T@M@T
        transformed=T@torch.linalg.solve(changed.T@changed+damping*changed_metric,-changed.T@r)
        coordinate_error=float((step-transformed).norm()/step.norm());assert coordinate_error<1e-9
        ordinary=torch.linalg.solve(A.T@A+damping*torch.eye(offset),-A.T@r)
        difference=float((A@(ordinary-step)).norm()/((A@step).norm()))
        rows.append(dict(seed=seed,damping=damping,relative_occurrence_step_error=error,
            relative_coordinate_error=coordinate_error,relative_prediction_difference_isotropic_damping=difference))
root=Path(__file__).parent;out=root/'results/metric_gauss_newton';out.mkdir(exist_ok=True)
write(out/'checks.json',dict(status='passed',rows=rows,source_sha256=digest(Path(__file__)),dependency_sha256=digest(root/'experiment.py'),
    scope='local linearized least squares on11probes; no training or convergence/generalization claim'))
print(rows)

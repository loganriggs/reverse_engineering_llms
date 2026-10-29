"""Check root-only normalization equivalence at epsilon zero; quantify finite-epsilon deviations."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from vision_circuits import BilinearCircuit
from run_svhn_standard import digest, write


def evaluate(model, x, eps, all_nodes):
    def norm(z):
        return z / (z.square().mean(-1, keepdim=True) + eps).sqrt()
    def combine(left, right, i, is_root):
        z = (left @ model.A[i].T) * (right @ model.B[i].T)
        return norm(z) if all_nodes or is_root else z
    if model.tree:
        def visit(i):
            if model.levels[i] == 0:
                return combine(x, x, i, i == 0)
            return combine(visit(2*i+1), visit(2*i+2), i, i == 0)
        h = visit(0)
    else:
        h = x
        for i in range(model.depth):
            h = combine(h, h, i, i == model.depth-1)
    return h @ model.W.T


def main():
    torch.set_num_threads(1)
    root = Path(__file__).parent
    out = root/'results/rms_collapse'
    out.mkdir(exist_ok=True)
    fixtures = []
    for depth in [2,3,4]:
        for tree in [False,True]:
            model = BilinearCircuit(5,4,depth,3,tree,True,201002,torch.float64)
            x = torch.randn(11,5,generator=torch.Generator().manual_seed(201003),dtype=torch.float64)
            a = evaluate(model,x,0.,True)
            b = evaluate(model,x,0.,False)
            torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
            target = torch.randn(a.shape,generator=torch.Generator().manual_seed(201004),dtype=torch.float64)
            ga = torch.autograd.grad((a*target).sum(),tuple(model.parameters()))
            gb = torch.autograd.grad((b*target).sum(),tuple(model.parameters()))
            for u,v in zip(ga,gb):
                torch.testing.assert_close(u,v,atol=1e-9,rtol=1e-9)
            fixtures.append(dict(depth=depth,tree=tree,max_output_error=float((a-b).abs().max().detach()),
                relative_gradient_error=max(float((u-v).norm()/u.norm().clamp_min(1e-12)) for u,v in zip(ga,gb))))
    observations = []
    scalar = BilinearCircuit(1,1,2,1,False,True,0,torch.float64)
    with torch.no_grad():
        for parameter in scalar.parameters():
            parameter.fill_(1.)
        tiny = torch.tensor([[1e-3]],dtype=torch.float64)
        every = float(evaluate(scalar,tiny,1e-6,True))
        root_only = float(evaluate(scalar,tiny,1e-6,False))
    assert every/root_only > 1e5
    counterexample = dict(input=.001, epsilon=1e-6, all_nodes=every, root_only=root_only,
                         ratio=every/root_only, purpose='finite epsilon is not an exact invariance')
    for regime in ['standard','extra']:
        for architecture in ['stack','tree']:
            directory = root/f'results/svhn_gpu/{regime}/d2_{architecture}'
            p = json.loads((directory/'protocol.json').read_text())
            images = np.load(p['manifests']['train']['images']['path'],mmap_mode='r')
            ids = np.load(directory/'splits.npz')['validation']
            x = images[ids].astype(np.float64)/255
            x = (x-np.array(p['mean'])[None,:,None,None])/np.array(p['std'])[None,:,None,None]
            model = BilinearCircuit(3072,32,2,10,architecture=='tree',True,200100,torch.float64)
            cp = directory/'checkpoint.pt'
            model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model'])
            chunks = [[],[]]
            with torch.no_grad():
                for start in range(0,5000,256):
                    xx = torch.from_numpy(x[start:start+256].reshape(-1,3072))
                    for j,all_nodes in enumerate([True,False]):
                        chunks[j].append(evaluate(model,xx,1e-6,all_nodes))
                a,b = (torch.cat(c) for c in chunks)
            observations.append(dict(regime=regime,architecture=architecture,checkpoint_sha256=digest(cp),
                max_absolute_logit_difference=float((a-b).abs().max()),
                relative_logit_difference=float((a-b).norm()/a.norm()),
                changed_predictions=int((a.argmax(1)!=b.argmax(1)).sum())))
    write(out/'checks.json',dict(status='passed',fixtures=fixtures,finite_epsilon_observations=observations,
        finite_epsilon_counterexample=counterexample,
        source_sha256=digest(Path(__file__)),dependency_sha256=digest(root/'vision_circuits.py'),
        interpretation='exact equivalence only at epsilon zero and nonzero intermediates; finite epsilon checked descriptively on depth2'))
    print(json.dumps(dict(fixtures=fixtures,observations=observations),indent=2))


if __name__=='__main__':
    main()

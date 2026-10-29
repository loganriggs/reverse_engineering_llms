"""Exact RMSNorm differential checks and fixed-probe local conditioning measurements."""
import json
from pathlib import Path
import numpy as np
import torch
from vision_circuits import BilinearCircuit
from run_svhn_standard import digest, write

torch.set_num_threads(1)
root = Path(__file__).parent
out = root / 'results/rms_geometry'
out.mkdir(exist_ok=True)
rng = torch.Generator().manual_seed(201001)
checks = []
for eps in [1e-6, .1, 1.]:
    for amplitude in [0., .001, 1., 10.]:
        z = amplitude * torch.randn(5, generator=rng, dtype=torch.float64)
        s = (z.square().mean() + eps).sqrt()
        expected = torch.eye(5, dtype=z.dtype) / s - torch.outer(z, z) / (5*s**3)
        actual = torch.autograd.functional.jacobian(lambda a: a/(a.square().mean()+eps).sqrt(), z)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        # Matrix-vector identities independently check tangential and radial eigenvalues.
        torch.testing.assert_close(actual @ z, eps/s**3*z, atol=1e-12, rtol=1e-6)
        if amplitude:
            v = torch.randn(5, generator=rng, dtype=z.dtype)
            v -= z * (z @ v) / (z @ z)
            torch.testing.assert_close(actual @ v, v/s, atol=1e-12, rtol=1e-10)
        checks.append(dict(eps=eps, amplitude=amplitude, max_jacobian_error=float((actual-expected).abs().max())))

class Instrumented(BilinearCircuit):
    def combine(self, left, right, i):
        z = (left @ self.A[i].T) * (right @ self.B[i].T)
        self.observations[i] = z.square().mean(-1).cpu().numpy()
        return z / torch.sqrt(z.square().mean(-1, keepdim=True) + 1e-6)

rows = []
for regime in ['standard', 'extra']:
    for architecture in ['stack', 'tree']:
        directory = root / f'results/svhn_gpu/{regime}/d2_{architecture}'
        p = json.loads((directory / 'protocol.json').read_text())
        image_path = Path(p['manifests']['train']['images']['path'])
        assert digest(image_path) == p['manifests']['train']['images']['sha256']
        images = np.load(image_path, mmap_mode='r')
        ids = np.load(directory / 'splits.npz')['validation'][:512]
        x = images[ids].astype(np.float64)/255
        x = (x-np.array(p['mean'])[None,:,None,None])/np.array(p['std'])[None,:,None,None]
        model = Instrumented(3072, 32, 2, 10, architecture=='tree', True, 200100, torch.float64)
        checkpoint = directory/'checkpoint.pt'
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
        model.observations = {}
        with torch.no_grad():
            model(torch.from_numpy(x.reshape(512,3072)))
        nodes = []
        for i, r2 in sorted(model.observations.items()):
            attenuation = 1e-6/(r2+1e-6)
            nodes.append(dict(node=i, level=model.levels[i], radial_to_tangential_eigenvalue_quantiles=
                              np.quantile(attenuation,[0,.5,1]).tolist(),
                              local_condition_number_quantiles=np.quantile(1/attenuation,[0,.5,1]).tolist()))
        rows.append(dict(regime=regime, architecture=architecture, checkpoint_sha256=digest(checkpoint), nodes=nodes))
result = dict(status='passed', checks=checks, rows=rows, probe='first512 fixed validation indices, float64 evaluation',
              quantiles=[0,.5,1], interpretation='local normalization Jacobian only; not network Hessian or proof of optimization causality',
              source_sha256=digest(Path(__file__)), dependency_sha256=digest(root/'vision_circuits.py'))
write(out/'checks.json',result)
print(json.dumps(result,indent=2))

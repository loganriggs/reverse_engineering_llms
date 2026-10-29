"""Independent NumPy replay and weighted-readout gradient audit."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from deep_multitask import DeepMultiTask
from check_deep_multitask import direct
from directional_sampling import DirectionalMixture


def main():
    torch.set_num_threads(1);out=Path(__file__).parent/'results/coverage_training'
    r=json.loads((out/'results.json').read_text());assert r['status']=='complete' and len(r['rows'])==12
    worst=0.;graderr=0.
    for row in r['rows']:
        d,s,m=row['depth'],row['seed'],row['method'];base=f'd{d}_s{s}';prefix=f'{base}_{m}'
        init=torch.load(out/f'{base}_initial.pt',weights_only=True)
        teacher=DeepMultiTask(16,8,6,d,True,0);teacher.load_state_dict(init['teacher'])
        model=DeepMultiTask(16,8,6,d,True,0);model.load_state_dict(torch.load(out/f'{prefix}.pt',weights_only=True))
        ev=np.load(out/f'{prefix}_evaluation.npz');x=ev['inputs'];w=ev['weights']
        a=direct(teacher,x);b=direct(model,x)
        values=np.column_stack([(a*a).mean(axis=1),((a-b)**2).mean(axis=1)])*w[:,None]
        err=np.max(np.abs(values-ev['weighted_values']))/max(1.,np.max(np.abs(values)));assert err<1e-10;worst=max(worst,float(err))
        q=DirectionalMixture(ev['axes'],(32,128));np.testing.assert_allclose(w,np.exp(-q.log_density(x/4)),rtol=1e-10)
        batches=values.reshape(6,8192,2).mean(axis=1);np.testing.assert_allclose(batches,row['batches'],rtol=1e-10)
        np.testing.assert_allclose(batches.mean(axis=0)[1]/batches.mean(axis=0)[0],row['relative_mse'],rtol=1e-10)
        data=np.load(out/f'{prefix}_training.npz');x=data['inputs'];w=data['weights'];y=data['targets']
        np.testing.assert_allclose(direct(teacher,x),y,rtol=1e-9,atol=1e-9)
        pred=direct(model,x);loss=np.mean(((pred-y)**2).mean(axis=1)*w);np.testing.assert_allclose(loss,row['training_loss'],rtol=1e-10)
        # Grad_W of the explicitly weighted mean, without self normalization.
        tx=torch.from_numpy(x[:64]);tw=torch.from_numpy(w[:64]);ty=torch.from_numpy(y[:64])
        pred=model(tx);objective=((pred-ty).square().mean(dim=1)*tw).mean()
        grad,=torch.autograd.grad(objective,model.W)
        z=model.hidden(tx).detach().numpy();i,j=np.triu_indices(8);features=z[:,i]*z[:,j]
        expected=(2/(64*6))*(w[:64,None]*(pred.detach().numpy()-y[:64])).T@features
        err=np.max(np.abs(expected-grad.numpy()))/max(1.,np.max(np.abs(expected)));assert err<1e-11;graderr=max(graderr,float(err))
        cal=np.load(out/f'{base}_calibration.npz');np.testing.assert_allclose(cal['scales']**2,cal['batch_energies'].mean(axis=0),rtol=1e-12)
        assert row['failure'] is None and row['completed_steps']==600
        print('Audited',prefix,flush=True)
    audit=dict(status='passed',fits=12,max_relative_forward_error=worst,max_weighted_gradient_error=graderr,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results_sha256=hashlib.sha256((out/'results.json').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()

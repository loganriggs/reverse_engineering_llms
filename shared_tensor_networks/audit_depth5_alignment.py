"""Replay final evaluations independently; verify frozen-data objectives."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from deep_multitask import DeepMultiTask
from directional_sampling import DirectionalMixture
from check_deep_multitask import direct


def main():
    torch.set_num_threads(1);root=Path(__file__).parent;src=root/'results/coverage_training';out=root/'results/depth5_alignment'
    protocol=json.loads((out/'protocol.json').read_text());result=json.loads((out/'results.json').read_text());assert result['status']=='complete' and len(result['rows'])==9
    for name,digest in protocol['input_hashes'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==digest
    worst=0.
    for row in result['rows']:
        seed,name,ci=row['seed'],row['config'],row['config_index'];prefix=f's{seed}_{name}'
        saved=torch.load(src/f'd5_s{seed}_initial.pt',weights_only=True)
        teacher=DeepMultiTask(16,8,6,5,True,0);teacher.load_state_dict(saved['teacher'])
        model=DeepMultiTask(16,8,6,5,True,0);model.load_state_dict(torch.load(out/f'{prefix}.pt',weights_only=True))
        data=np.load(src/f'd5_s{seed}_importance_training.npz')
        pred=direct(model,data['inputs']);loss=np.mean(data['weights']*np.mean((pred-data['targets'])**2,axis=1));np.testing.assert_allclose(loss,row['trace'][-1]['training_loss'],rtol=1e-10,atol=1e-12)
        init=DeepMultiTask(16,8,6,5,True,0);init.load_state_dict(torch.load(out/f'{prefix}_initial.pt',weights_only=True))
        expected=dict(saved['initial']);k=int(name[-1]);cal=np.load(out/f'{prefix}_calibration.npz')
        np.testing.assert_allclose(cal['norm']**2,cal['batch_energies'].mean(axis=0),rtol=1e-12)
        for layer in range(k):
            for key in [f'A.{layer}',f'B.{layer}',f'scale_{layer}']:expected[key]=saved['teacher'][key]
        expected['W']=expected['W']/torch.from_numpy(cal['norm'][:,None])
        for key,value in init.state_dict().items():torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
        initial=direct(init,data['inputs']);loss=np.mean(data['weights']*np.mean((initial-data['targets'])**2,axis=1));np.testing.assert_allclose(loss,row['trace'][0]['training_loss'],rtol=1e-10,atol=1e-12)
        ev=np.load(out/f'{prefix}_evaluation.npz');q=DirectionalMixture(ev['axes'],(32,128));rng=np.random.default_rng(135000+10*seed+ci);chunks=[]
        for chunk in range(24):
            u,w=q.sample(4096,rng);x=4*u;yy=direct(teacher,x);pp=direct(model,x)
            v=np.column_stack([(yy*yy).mean(axis=1),((pp-yy)**2).mean(axis=1),(pp*pp).mean(axis=1),(pp*yy).mean(axis=1)])*w[:,None];chunks.append(v)
        values=np.concatenate(chunks);err=np.max(np.abs(values-ev['weighted_values']))/max(1.,np.max(np.abs(values)));assert err<1e-10;worst=max(worst,float(err))
        batches=values.reshape(6,16384,4).mean(axis=1);np.testing.assert_allclose(batches,row['batches'],rtol=1e-8,atol=1e-10)
        assert row['failure'] is None and row['completed_steps']==600
        print('Audited',prefix,flush=True)
    audit=dict(status='passed',fits=9,max_independent_forward_relative_error=worst,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results_sha256=hashlib.sha256((out/'results.json').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()

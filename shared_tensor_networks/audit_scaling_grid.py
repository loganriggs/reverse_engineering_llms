"""Independent forwards and transfer replay for the representative grid."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from deep_multitask import DeepMultiTask
from directional_sampling import DirectionalMixture
from check_deep_multitask import direct


def main():
    torch.set_num_threads(1);out=Path(__file__).parent/'results/scaling_grid'
    results=json.loads((out/'results.json').read_text());assert results['status']=='complete' and len(results['rows'])==72
    worst=0.;audited=0;failures=[]
    for row in results['rows']:
        if row.get('failure'):
            failures.append(row);continue
        folder=out/row['key'];d,h,depth,rep=row['d'],row['h'],row['depth'],row['replicate'];name=row['model']
        gi=row['grid_index'];ki=int(row['teacher']=='independent');mi=['shared','separate','shared_budget'].index(name)
        base=2000000+10000000*gi+1000000*ki+10000*rep
        teacher=DeepMultiTask(d,h,16,depth,ki==0,0);teacher.load_state_dict(torch.load(folder/'teacher.pt',weights_only=True))
        model=DeepMultiTask(d,row['width'],12,depth,name!='separate',0);model.load_state_dict(torch.load(folder/f'{name}.pt',weights_only=True))
        assert sum(p.numel() for p in model.parameters())==row['parameters']
        data=np.load(folder/'training.npz');pred=direct(model,data['inputs']);truth=direct(teacher,data['inputs'])[:,:12]
        np.testing.assert_allclose(truth,data['targets'],rtol=1e-8,atol=1e-8)
        loss=np.mean(((pred-truth)**2).mean(axis=1)*data['weights']);np.testing.assert_allclose(loss,row['train_mse'],rtol=1e-9,atol=1e-10)
        ev=np.load(folder/f'{name}_evaluation.npz');adapt=data['adaptation_inputs'];yy=direct(teacher,adapt)[:,12:]
        np.testing.assert_allclose(yy,data['adaptation_targets'],rtol=1e-8,atol=1e-8)
        for norm,predict,coeffs in [(ev['feature_norm'],lambda x:direct(model,x),ev['B']),(ev['oracle_norm'],lambda x:direct(teacher,x)[:,:12],ev['oracle_B'])]:
            f=predict(adapt)/norm
            for k,beta in zip([32,128],coeffs):
                residual=f[:k].T@(f[:k]@beta-yy[:k])/k+.001*beta
                assert np.linalg.norm(residual)/max(1.,np.linalg.norm(f[:k].T@yy[:k]/k))<1e-9
        eq=DirectionalMixture(ev['axes'],(16,64));batches=[]
        for batch in range(6):
            u,w=eq.sample(8192,np.random.default_rng(base+50+10*mi+batch));x=u*math.sqrt(d)
            t=direct(teacher,x);s=direct(model,x);f=s/ev['feature_norm'];oracle=t[:,:12]/ev['oracle_norm']
            prediction=np.column_stack([s]+[f@b for b in ev['B']]+[oracle@b for b in ev['oracle_B']])
            target=np.column_stack([t[:,:12]]+[t[:,12:]]*4)
            delta=(prediction-target)**2;energy=target**2
            values=np.column_stack([delta[:,:12].mean(axis=1),energy[:,:12].mean(axis=1),delta[:,12:],energy[:,12:]])*w[:,None]
            saved=ev['weighted_values'][batch*8192:(batch+1)*8192]
            error=np.max(np.abs(values-saved))/max(1.,np.max(np.abs(values)));assert error<1e-9;worst=max(worst,float(error));batches.append(values.mean(axis=0))
        np.testing.assert_allclose(batches,row['batches'],rtol=1e-8,atol=1e-9)
        assert row['completed_steps']==600;audited+=1;print('Audited',row['key'],name,flush=True)
    report=dict(status='passed',audited_fits=audited,retained_failures=failures,max_independent_relative_forward_error=worst,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results_sha256=hashlib.sha256((out/'results.json').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()

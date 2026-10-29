"""M74 all full-vocabulary outputs independently rescored and residuals reconstructed."""
import json
import numpy as np
import torch
from downstream_metric_helpers import P,V,sha
from audit_downstream_metric_preflight import energy

def lp(x):
    z=np.asarray(x,dtype=np.float64);z=z-z.max(-1,keepdims=True)
    return z-np.log(np.exp(z).sum(-1,keepdims=True))

def main():
    torch.set_num_threads(1);rp=V/'downstream_metric_results.json';result=json.loads(rp.read_text());execution=json.loads((V/'downstream_metric_execution.json').read_text())
    assert result['execution_sha256']==sha(V/'downstream_metric_execution.json')
    assert result['source_sha256']==sha(P/'probe_downstream_metric.py') and result['helper_sha256']==sha(P/'downstream_metric_helpers.py')
    assert result['preflight_audit_sha256']==sha(V/'downstream_metric_preflight_audit.json')
    references=json.loads((V/'downstream_residual_metric_protocol.json').read_text())['references']
    for name,digest in references.items():assert sha(V/name)==digest
    cohort=json.loads((V/'natural_token_context_cohort.json').read_text())
    fit=json.loads((V/'pca_encoder_fit.json').read_text());assert sha(fit['state_path'])==fit['state_sha256'];pca=np.load(fit['state_path'])
    decoders=[]
    for e in execution['models']:
        if e['kind']=='pca':u=pca['basis'][:,:128];decoders.append((pca['mean'],u@u.T))
        else:
            assert sha(e['path'])==e['sha256'];s=torch.load(e['path'],weights_only=True)
            D=s['state']['dictionary'].double().numpy();D/=np.linalg.norm(D,axis=0)
            u,sv,_=np.linalg.svd(D,full_matrices=False);keep=sv>1e-10*sv[0];u=u[:,keep]
            center=s['mean'].double().numpy()+float(s['scale'])*s['state']['bias'].double().numpy()
            decoders.append((center,u@u.T))
    files={(f['document'],f['model_index']):f for f in result['files']};assert len(files)==640
    rows={(r['document'],r['kind'],r['seed'],r['arm']):r for r in result['rows']};assert len(rows)==608
    maxerr=0.;maxres=0.;vectors=0;cases=0
    def check(a,b):
        nonlocal maxerr
        assert np.isclose(a,b,rtol=1e-8,atol=1e-9),(a,b);maxerr=max(maxerr,abs(a-b))
    for doc in execution['evaluation_documents']:
        nf=files[doc,None];assert sha(nf['path'])==nf['sha256'];native=torch.load(nf['path'],weights_only=True)
        assert native['ids'][0].tolist()==cohort['token_ids'][doc];base=native['native'].numpy();prob_log=lp(base);prob=np.exp(prob_log);w=native['write'].double().numpy()
        for i,e in enumerate(execution['models']):
            f=files[doc,i];assert sha(f['path'])==f['sha256'];s=torch.load(f['path'],weights_only=True)
            assert s['document']==doc and all(s['model'][k]==e[k] for k in ['kind','seed','arm'])
            row=rows[doc,e['kind'],e['seed'],e['arm']];res=s['residual'].double().numpy();center,projection=decoders[i]
            reference=center+(w-center)@projection-w;err=float(np.linalg.norm(reference-res)/np.linalg.norm(res));assert err<1e-5;maxres=max(maxres,err)
            check(float(np.square(res).sum()),row['residual_energy']);check(float(np.square(res).sum()/np.square(w).sum()),row['relative_residual_energy'])
            E=energy(base,s['tangent'].numpy());check(E,row['Fisher_energy'])
            assert len(s['outputs'])==len(row['finite'])==3
            for output,metric,alpha in zip(s['outputs'],row['finite'],execution['amplitudes']):
                assert output['alpha']==metric['alpha']==alpha
                kl=float((prob*(prob_log-lp(output['logits'].numpy()))).sum(-1).mean());pred=.5*alpha**2*E
                check(kl,metric['KL']);check(pred,metric['Taylor_KL']);check(kl/pred,metric['actual_over_Taylor']);check(abs(kl-pred),metric['Taylor_absolute_error']);vectors+=1
            check(abs(row['finite'][-1]['KL']-row['prior_full_KL']),row['prior_KL_replay_absolute_error']);cases+=1
        print('AUDIT_DOC',doc,flush=True)
    assert cases==608 and vectors==1824 and result['forward_count']==2464
    out=dict(status='passed',result_sha256=sha(rp),source_sha256=sha(__file__),Fisher_metric_source_sha256=sha(P/'audit_downstream_metric_preflight.py'),cases=cases,full_output_vectors=vectors,max_metric_absolute_error=maxerr,max_independent_residual_relative_error=maxres,scope='All608 saved tangents and1824full outputs scored independently in NumPy; all608 residuals independently reconstructed from native writes and frozen projection spaces. Derivative validity additionally established by fixed38case preflight, not finite differences on every full-evaluation case.')
    op=V/'downstream_metric_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()

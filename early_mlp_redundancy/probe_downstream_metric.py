"""M74 fixed residual directions: local Fisher predictions versus finite KL."""
import json
import torch
from inspect_model import load_model,trace
from downstream_residual_metric import directional_logits,perturbed_logits,fisher_energy
from downstream_metric_helpers import P,V,A,setup,identity,sha

def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
    execution,encoders,cohort,writes=setup()
    check=json.loads((V/'downstream_metric_preflight.json').read_text());assert check['status']=='passed' and check['execution_sha256']==sha(V/'downstream_metric_execution.json')
    assert check['helper_sha256']==sha(P/'downstream_metric_helpers.py')
    audit=json.loads((V/'downstream_metric_preflight_audit.json').read_text());assert audit['status']=='passed' and audit['result_sha256']==sha(V/'downstream_metric_preflight.json')
    prior={}
    for name in ['pca_natural.json','least_squares_natural.json']:
        for r in json.loads((V/name).read_text())['rows']:prior[r['kind'],r['seed'],r['arm'],r['document']]=r
    model,receipt=load_model();model.requires_grad_(False)
    dest=A/'emlp_m74_metric_samples';dest.mkdir(exist_ok=True)
    progress=A/'emlp_m74_metric_progress.pt';op=V/'downstream_metric_results.json';assert not op.exists()
    rows=[];files=[];calls=0;completed=0
    bindings=dict(execution_sha256=sha(V/'downstream_metric_execution.json'),source_sha256=sha(__file__),helper_sha256=sha(P/'downstream_metric_helpers.py'),preflight_audit_sha256=sha(V/'downstream_metric_preflight_audit.json'))
    if progress.exists():
        saved=torch.load(progress,weights_only=True)
        for k,value in bindings.items():assert saved[k]==value
        rows,files,calls,completed=[saved[k] for k in ['rows','files','calls','completed']]
        for f in files:assert sha(f['path'])==f['sha256']
    for di,doc in enumerate(execution['evaluation_documents']):
        if di<completed:continue
        ids=torch.tensor([cohort['token_ids'][doc]],device='cuda');base,cap=trace(model,ids);calls+=1
        w=model.transformer.h[1].mlp.Down(cap[1]['products']);assert torch.equal(w[0].cpu(),writes[doc*128:(doc+1)*128])
        lp=base.double().log_softmax(-1);prob=lp.exp()
        native_path=dest/f'doc{doc}_native.pt';torch.save(dict(ids=ids.cpu(),native=base.cpu(),write=w.cpu()),native_path)
        files.append(dict(document=doc,model_index=None,path=str(native_path),sha256=sha(native_path)))
        for i,(e,enc) in enumerate(encoders):
            rec,*_=enc.reconstruct(w,e['variant']);residual=rec-w
            primal,tangent=directional_logits(model,ids,residual);calls+=1;assert torch.equal(primal,base)
            energy=fisher_energy(base,tangent);F=float(energy.mean());finite=[];outputs=[]
            for alpha in execution['amplitudes']:
                out=perturbed_logits(model,ids,residual,alpha);calls+=1
                kl=(prob*(lp-out.double().log_softmax(-1))).sum(-1);actual=float(kl.mean());predicted=.5*alpha**2*F
                finite.append(dict(alpha=alpha,KL=actual,Taylor_KL=predicted,actual_over_Taylor=actual/predicted,Taylor_absolute_error=abs(actual-predicted)))
                outputs.append(dict(alpha=alpha,logits=out.cpu()))
            previous=prior[e['kind'],e['seed'],e['arm'],doc]['mean_KL'];replay_error=abs(finite[-1]['KL']-previous)
            assert replay_error<=execution['prior_KL_replay_atol']+execution['prior_KL_replay_rtol']*abs(previous),(doc,e,replay_error)
            row=identity(e)|dict(document=doc,Fisher_energy=F,residual_energy=float(residual.double().square().sum()),relative_residual_energy=float(residual.double().square().sum()/w.double().square().sum()),prior_full_KL=previous,prior_KL_replay_absolute_error=replay_error,finite=finite)
            path=dest/f'doc{doc}_model{i}.pt';torch.save(dict(document=doc,model=identity(e),residual=residual.cpu(),tangent=tangent.cpu(),outputs=outputs),path)
            rows.append(row);files.append(dict(document=doc,model_index=i,path=str(path),sha256=sha(path)))
        temp=progress.with_suffix('.tmp');torch.save(bindings|dict(rows=rows,files=files,calls=calls,completed=di+1),temp);temp.replace(progress)
        print('DOCS',di+1,'FORWARDS',calls,flush=True)
    assert len(rows)==608 and len(files)==640 and calls==2464
    result=bindings|dict(status='completed',checkpoint_sha256=receipt.weights_sha256,forward_count=calls,rows=rows,files=files)
    op.write_text(json.dumps(result,indent=2)+'\n');print('DONE',calls,flush=True)
if __name__=='__main__':main()

"""M77 native captures and analytic predictions, before downstream scoring."""
import json
import torch
from inspect_model import load_model,trace
from native_input_operator import predictions
from fit_pca_encoder import sha,V,A
P=V.parent

def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
    ep=V/'native_input_operator_execution.json';execution=json.loads(ep.read_text());assert execution['operator_source_sha256']==sha(P/'native_input_operator.py')
    ref=json.loads((V/'native_input_reference.json').read_text());assert ref['execution_sha256']==sha(ep) and sha(ref['path'])==ref['sha256'];state=torch.load(ref['path'],weights_only=True);L,R,D,mean=[state[k].double() for k in ['L','R','D','mean']]
    protocol=json.loads((V/'native_input_operator_protocol.json').read_text())
    for name,digest in protocol['references'].items():assert sha(V/name)==digest
    q=json.loads((V/'whole_task_native_protocol.json').read_text());native=json.loads((V/'whole_task_native.json').read_text());assert sha(native['samples_path'])==native['samples_sha256'];saved=torch.load(native['samples_path'],weights_only=True)
    model,receipt=load_model();mlp=model.transformer.h[1].mlp
    for key,actual in [('L',mlp.Left.weight),('R',mlp.Right.weight),('D',mlp.Down.weight)]:assert torch.equal(state[key],actual.cpu())
    dest=A/'emlp_m77_input_changes';dest.mkdir(exist_ok=True);rows=[];files=[];calls=0;max_mid=0.
    op=V/'native_input_changes.json';assert not op.exists()
    for pi,pair in enumerate(q['pairs']):
        zs=[];writes=[]
        for endpoint in (0,1):
            ids=torch.tensor([pair['token_ids'][endpoint]],device='cuda');out,cap=trace(model,ids);calls+=1
            assert torch.equal(out[0,-1].cpu(),saved['baselines'][pair['id']+f'|{endpoint}']['logits'])
            zs.append(cap[1]['normalized_input'].cpu());writes.append(mlp.Down(cap[1]['products']).cpu())
        entries=[]
        for target in [t for t in native['records'] if t['pair_id']==pair['id']]:
            if target['status']!='evaluated':continue
            endpoint=target['endpoint'];pos=target['recipient_positions'];dp=target['donor_positions'];zr=zs[endpoint][:,pos].double();zd=zs[1-endpoint][:,dp].double();nd=writes[1-endpoint][:,dp]-writes[endpoint][:,pos]
            assert torch.equal(nd,saved['deltas'][target['sample_key']]['delta']);pred=predictions(L,R,D,mean,zr,zd);cast={k:x.float() for k,x in pred.items()};energy=float(nd.double().square().sum())
            err=float((cast['midpoint_J_exact_control'].double()-nd.double()).square().sum());relative=err/max(energy,1e-30);max_mid=max(max_mid,relative);assert relative<=execution['midpoint_local_relative_squared_tolerance']
            for name,value in cast.items():
                rows.append(dict(pair_id=pair['id'],endpoint=endpoint,family=pair['family'],format=pair['format'],region=target['region'],predictor=name,target_delta_energy=energy,predicted_delta_energy=float(value.double().square().sum()),delta_error=float((value.double()-nd.double()).square().sum()),relative_delta_error=float((value.double()-nd.double()).square().sum())/energy if energy else None))
            entries.append(dict(sample_key=target['sample_key'],endpoint=endpoint,region=target['region'],recipient_positions=pos,donor_positions=dp,predictions=cast))
        path=dest/f'pair_{pi}.pt';torch.save(dict(pair_id=pair['id'],z=zs,writes=writes,entries=entries),path);files.append(dict(pair_id=pair['id'],path=str(path),sha256=sha(path)))
        if (pi+1)%8==0:print('LOCAL_PAIRS',pi+1,flush=True)
    assert len(rows)==984 and calls==112
    out=dict(status='local_preflight_passed',execution_sha256=sha(ep),source_sha256=sha(__file__),reference_sha256=sha(V/'native_input_reference.json'),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,rows=rows,files=files,max_midpoint_relative_squared_error=max_mid,ineligible=[x for x in native['records'] if x['status']!='evaluated'])
    op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls,max_mid)
if __name__=='__main__':main()

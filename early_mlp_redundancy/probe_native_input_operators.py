"""M77 downstream effects of fixed native input-to-write change predictions."""
import json
import torch
from inspect_model import load_model,trace
from fit_pca_encoder import sha,V,A
from probe_source_holdout_native import margin
P=V.parent

def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
    local=json.loads((V/'native_input_changes.json').read_text());audit=json.loads((V/'native_input_changes_audit.json').read_text());assert audit['status']=='passed' and audit['result_sha256']==sha(V/'native_input_changes.json')
    execution=json.loads((V/'native_input_operator_execution.json').read_text());assert local['execution_sha256']==sha(V/'native_input_operator_execution.json')
    protocol=json.loads((V/'whole_task_native_protocol.json').read_text());pairs={p['id']:p for p in protocol['pairs']}
    native=json.loads((V/'whole_task_native.json').read_text());assert sha(native['samples_path'])==native['samples_sha256'];saved=torch.load(native['samples_path'],weights_only=True);nrows={x['sample_key']:x for x in native['records']}
    model,receipt=load_model();rows=[];files=[];calls=0;teacher_replays=0;dest=A/'emlp_m77_native_operator_samples';dest.mkdir(exist_ok=True);op=V/'native_input_operator_results.json';assert not op.exists()
    for fi,file in enumerate(local['files']):
        assert sha(file['path'])==file['sha256'];local_sample=torch.load(file['path'],weights_only=True);pair=pairs[file['pair_id']];cs=[];blob=dict(pair_id=pair['id'],changed={})
        for ep in (0,1):
            ids=torch.tensor([pair['token_ids'][ep]],device='cuda');out,cap=trace(model,ids);calls+=1;base=out[0,-1];key=pair['id']+f'|{ep}';assert torch.equal(base.cpu(),saved['baselines'][key]['logits']);write=model.transformer.h[1].mlp.Down(cap[1]['products']);assert torch.equal(write.cpu(),local_sample['writes'][ep]);cs.append(dict(ids=ids,base=base,write=write,key=key))
        for entry in local_sample['entries']:
            ep=entry['endpoint'];c=cs[ep];target=nrows[entry['sample_key']];pos=entry['recipient_positions'];nd=torch.zeros_like(c['write']);nd[:,pos]=saved['deltas'][entry['sample_key']]['delta'].cuda()
            if bool(nd.count_nonzero()):out,_=trace(model,c['ids'],replace=lambda j,z,w:w+nd if j==1 else w);calls+=1;teacher=out[0,-1]
            else:teacher=c['base']
            assert torch.equal(teacher.cpu(),saved['changed'][entry['sample_key']]);teacher_replays+=1
            blp=c['base'].double().log_softmax(-1);response=teacher.double().log_softmax(-1)-blp;g=pair['answer_groups'][ep];variant=pair['variants'][ep]
            for predictor,pred in entry['predictions'].items():
                delta=torch.zeros_like(nd);delta[:,pos]=pred.cuda()
                if bool(delta.count_nonzero()):out,_=trace(model,c['ids'],replace=lambda j,z,w:w+delta if j==1 else w);calls+=1;changed=out[0,-1]
                else:changed=c['base']
                cr=changed.double().log_softmax(-1)-blp;error=float((blp.exp()*(cr-response).square()).sum());effect=margin(changed,g,variant)-margin(c['base'],g,variant)
                if predictor=='midpoint_J_exact_control':assert error<=execution['midpoint_response_error_atol']+execution['midpoint_response_error_rtol']*target['response_energy'],(pair['id'],entry['region'],error,target['response_energy'])
                rows.append(dict(pair_id=pair['id'],key=c['key'],endpoint=ep,family=pair['family'],format=pair['format'],region=entry['region'],predictor=predictor,response_error=error,target_energy=target['response_energy'],margin_error=(effect-target['margin_change'])**2,margin_target_energy=target['margin_change']**2,delta_energy=float(delta.double().square().sum()),noop=not bool(delta.count_nonzero())))
                blob['changed'][entry['sample_key']+'|'+predictor]=changed.cpu()
        path=dest/f'pair{fi}.pt';torch.save(blob,path);files.append(dict(pair_id=pair['id'],path=str(path),sha256=sha(path)))
        if (fi+1)%8==0:print('RESPONSE_PAIRS',fi+1,'FORWARDS',calls,flush=True)
    assert len(rows)==984 and teacher_replays==328 and calls==976
    out=dict(status='completed',source_sha256=sha(__file__),local_sha256=sha(V/'native_input_changes.json'),local_audit_sha256=sha(V/'native_input_changes_audit.json'),execution_sha256=sha(V/'native_input_operator_execution.json'),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,teacher_replays=teacher_replays,rows=rows,files=files,ineligible=[x|dict(predictor=name) for x in local['ineligible'] for name in execution['predictors']])
    op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls)
if __name__=='__main__':main()

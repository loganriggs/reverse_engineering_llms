"""M33 fixed two-scale symmetric native interventions; resumable, no fitting."""
from pathlib import Path
import hashlib
import json
import torch
from inspect_model import load_model, trace
P = Path(__file__).resolve().parent / 'validation'
A = Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    pp = P/'native_local_protocol.json'
    p = json.loads(pp.read_text())
    rp = P/'native_response_geometry.json'
    assert sha(rp) == p['parent_result_sha256']
    r = json.loads(rp.read_text())
    vp = Path(r['vectors_path'])
    assert sha(vp) == r['vectors_sha256']
    parent = torch.load(vp, weights_only=True)
    ep = P/'context_expansion_protocol.json'
    assert sha(ep) == p['expansion_sha256']
    expanded = {x['id']: x for x in json.loads(ep.read_text())['rows']}
    model, receipt = load_model()
    assert receipt.weights_sha256 == r['checkpoint_sha256']
    device = next(model.parameters()).device
    op = P/'native_local_inventory.json'
    assert not op.exists(), 'Completed result exists; do not duplicate the experiment.'
    progress = A/'emlp_m33_native_local_progress.pt'
    records, baselines, calls = [], {}, 0
    if progress.exists():
        saved = torch.load(progress, weights_only=True)
        assert saved['protocol_sha256'] == sha(pp)
        records, baselines, calls = saved['records'], saved['baselines'], saved['calls']
    done = {(x['target_index'],x['context']) for x in records}
    assert len(done) == len(records)
    for n,s in enumerate(p['rows']):
        key = s['target_index'], s['context']
        if key in done: continue
        nativekey = s['pair_id'], s['endpoint']
        pair = expanded[s['pair_id']]
        ids = torch.tensor([pair['token_ids'][s['endpoint']]],device=device)
        boundary = s['prefix_length']
        assert ids[0,boundary:].tolist() == s['task_token_ids']
        if nativekey not in baselines:
            out,_ = trace(model,ids)
            intact = out[0,-1].cpu().clone()
            assert torch.equal(intact,parent['nativecache'][nativekey]), 'Baseline did not exactly replay.'
            baselines[nativekey] = intact
            calls += 1
        taskdelta = parent['deltas'][s['target_index']].to(device)
        delta = torch.zeros((1,ids.shape[1],taskdelta.shape[-1]),device=device,dtype=taskdelta.dtype)
        delta[:,boundary:] = taskdelta
        noop = not bool(torch.count_nonzero(delta))
        vectors = {}
        for h in p['amplitudes']:
            for sign in [-1,1]:
                label = f'{h}:{sign}'
                if noop: vectors[label] = baselines[nativekey].clone()
                else:
                    # Positive amplitude follows the M32 removal direction.
                    out,_ = trace(model,ids,replace=lambda j,z,w:w-sign*h*delta if j==1 else w)
                    vectors[label] = out[0,-1].cpu().clone()
                    calls += 1
        records.append(dict(target_index=key[0],context=key[1],kind=s['kind'],seed=s['seed'],family=s['family'],native_key=nativekey,noop=noop,task_delta_sha256=hashlib.sha256(taskdelta.cpu().numpy().tobytes()).hexdigest(),vectors=vectors))
        if len(records)%30==0 or n==len(p['rows'])-1:
            temp = progress.with_suffix('.tmp')
            torch.save(dict(protocol_sha256=sha(pp),records=records,baselines=baselines,calls=calls),temp)
            temp.replace(progress)
            print('SAVED',len(records),'/',len(p['rows']),'FORWARDS',calls,flush=True)
    output = A/'emlp_m33_native_local_vectors.pt'
    progress.replace(output)
    result = dict(protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),parent_result_sha256=sha(rp),checkpoint_sha256=receipt.weights_sha256,records=len(records),unique_inputs=len(baselines),noops=sum(x['noop'] for x in records),forward_count=calls,vectors_path=str(output),vectors_sha256=sha(output))
    assert result['records']==720 and result['noops']==93
    assert calls==len(baselines)+4*(len(records)-result['noops'])
    op.write_text(json.dumps(result,indent=2)+'\n')
    print('DONE',result,flush=True)
if __name__=='__main__': main()

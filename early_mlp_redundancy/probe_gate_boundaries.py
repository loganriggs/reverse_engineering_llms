"""M110 exact Top8 support boundaries; downstream response to one-sided limits."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,A,sha
from block_gate_boundaries import boundaries,decoded,support
from inspect_model import load_model,trace


def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
    pp=V/'gate_boundary_protocol.json';p=json.loads(pp.read_text());assert p['source_sha256']==sha(__file__)
    for name,digest in p['bindings'].items():assert sha(V/name)==digest
    assert sha(V.parent/'block_gate_boundaries.py')==p['helper_sha256']
    cache=json.loads((V/'block_change_roles_whole.json').read_text());pairs=json.loads((V/'whole_task_native_protocol.json').read_text())['pairs'];byid={x['id']:x for x in pairs}
    states=[]
    for e in p['models']:
        assert sha(e['path'])==e['sha256'];s=torch.load(e['path'],weights_only=True);states.append((e,{k:t.double().numpy() for k,t in s['state'].items()},s['mean'].double().numpy(),s['scale'].double().numpy()))
    dest=V/'gate_boundary_native.json';sp=A/'emlp_m110_gate_boundary_samples.pt';assert not dest.exists() and not sp.exists()
    model,mr=load_model();nr=json.loads((V/'whole_task_native.json').read_text());assert mr.weights_sha256==nr['checkpoint_sha256']
    results=[];samples={};calls=0;maxlimit=0.
    for entry in cache['files']:
      assert sha(entry['path'])==entry['sha256'];data=torch.load(entry['path'],weights_only=True);pair=byid[entry['pair_id']];pid=pair['id']
      w0=data['writes'][0][0,-1].double().numpy();w1=data['writes'][1][0,-1].double().numpy();ids=torch.tensor([pair['token_ids'][0]],device='cuda');baseline,cap=trace(model,ids);calls+=1;assert torch.equal(baseline[0,-1].cpu(),data['baselines'][0])
      for e,s,mu,scale in states:
        D=s['dictionary']/np.linalg.norm(s['dictionary'],axis=0)
        a=(((w0-mu)/scale)@s['encoder'].T+s['encoder_bias']).reshape(32,4)
        b=(((w1-w0)/scale)@s['encoder'].T).reshape(32,4)
        rr,ties=boundaries(a,b,8);assert not ties,'permanent tie requires explicit handling'
        key=f"{pid}|{e['seed']}|{e['arm']}";row=dict(pair_id=pid,family=pair['family'],seed=e['seed'],arm=e['arm'],boundary_count=len(rr),boundaries=rr,native_path_norm=float(np.linalg.norm(w1-w0)))
        if not rr:row['status']='no_crossing';results.append(row);continue
        r=rr[0];t=r['t'];left=decoded(a,b,t,D,r['left'])*scale+s['bias']*scale+mu;right=decoded(a,b,t,D,r['right'])*scale+s['bias']*scale+mu;teacher=w0+t*(w1-w0);jump=right-left
        finite=[]
        for f in p['fractions']:
          h=r['radius']*f
          assert list(support(a,b,t-h,8))==r['left'] and list(support(a,b,t+h,8))==r['right']
          actual=(decoded(a,b,t+h,D,r['right'])-decoded(a,b,t-h,D,r['left']))*scale
          slope=(decoded(np.zeros_like(a),b,1,D,r['right'])+decoded(np.zeros_like(a),b,1,D,r['left']))*scale
          err=float(np.linalg.norm(actual-jump-h*slope)/max(np.linalg.norm(jump),1))
          maxlimit=max(maxlimit,err);assert err<1e-9
          finite.append(dict(fraction=f,h=h,native_change_norm=float(2*h*np.linalg.norm(w1-w0)),decoded_change_norm=float(np.linalg.norm(actual)),limit_error_norm=float(np.linalg.norm(actual-jump))))
        outputs={}
        for mode,write in [('left',left),('right',right),('native_path',teacher)]:
          w=torch.tensor(write,dtype=torch.float32,device='cuda')
          def replace(i,z,y):
            if i!=1:return y
            assert torch.equal(z,cap[1]['normalized_input']);out=y.clone();out[0,-1]=w+model.transformer.h[1].mlp.Down_bias;assert torch.equal(out[:,:-1],y[:,:-1]);return out
          logits,_=trace(model,ids,replace=replace);calls+=1;outputs[mode]=logits[0,-1].cpu()
        row.update(status='evaluated',first_boundary=t,jump_norm=float(np.linalg.norm(jump)),jump_over_native_path=float(np.linalg.norm(jump)/np.linalg.norm(w1-w0)),finite_limits=finite)
        samples[key]=dict(left_write=torch.from_numpy(left),right_write=torch.from_numpy(right),native_path_write=torch.from_numpy(teacher),outputs=outputs,a=torch.from_numpy(a),b=torch.from_numpy(b))
        results.append(row)
      print('PAIR',pid,'FORWARDS',calls,flush=True)
    torch.save(samples,sp);out=dict(status='completed_pending_audit',source_sha256=sha(__file__),protocol_sha256=sha(pp),checkpoint_sha256=mr.weights_sha256,forward_count=calls,baseline_replays=len(cache['files']),max_local_limit_identity_error=maxlimit,rows=results,path=str(sp),sha256=sha(sp));dest.write_text(json.dumps(out,indent=2)+'\n');print('COMPLETED',calls,'PATHS',len(results),'CROSSING',len(samples),flush=True)
if __name__=='__main__':main()

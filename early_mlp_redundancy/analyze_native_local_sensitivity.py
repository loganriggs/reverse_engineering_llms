"""M33 fixed local/full response comparison, all rows retained."""
from pathlib import Path
import hashlib,json,math,statistics
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def energy(x,w):return float(np.dot(w,x*x))
def center(x,w):return x-np.dot(w,x)
def ratio(a,b):return a/b if b else None
def geometry(u,v,w):
    er,ec=energy(u,w),energy(v,w)
    diff=energy(v-u,w)
    dot=float(np.dot(w,u*v))
    return dict(reference_energy=er,context_energy=ec,difference_energy=diff,cosine=dot/math.sqrt(er*ec) if er and ec else None,orthogonal_energy=max(0.,ec-dot*dot/er) if er else ec)
def main():
    ip=P/'native_local_inventory.json';inv=json.loads(ip.read_text());pp=P/'native_local_protocol.json';p=json.loads(pp.read_text());assert inv['protocol_sha256']==sha(pp)
    vp=Path(inv['vectors_path']);assert sha(vp)==inv['vectors_sha256'];data=torch.load(vp,weights_only=True)
    rp=P/'native_response_geometry.json';parent=json.loads(rp.read_text());assert sha(rp)==p['parent_result_sha256'];pv=Path(parent['vectors_path']);assert sha(pv)==parent['vectors_sha256'];full=torch.load(pv,weights_only=True)
    lookup={(r['target_index'],r['context']):r for r in data['records']};fl={(r['target_index'],r['context']):r for r in full['vectors']};weights={};responses={};rows=[]
    for x in data['records']:
        i,c=x['target_index'],x['context'];ref=lookup[i,'reference'];base=data['baselines'][ref['native_key']].numpy().astype(float)
        if i not in weights:
            w=np.exp(base-base.max());w/=w.sum();weights[i]=w
        w=weights[i];ds={};resolution={}
        for h in p['amplitudes']:
            plus=x['vectors'][f'{h}:1'].numpy();minus=x['vectors'][f'{h}:-1'].numpy();raw=plus.astype(float)-minus.astype(float)
            ds[str(h)]=center(raw/(2*h),w)
            ulps=np.abs(np.spacing(plus)).astype(float)+np.abs(np.spacing(minus)).astype(float)
            resolution[str(h)]=math.sqrt(energy(raw,w)/energy(ulps,w)) if energy(ulps,w) else None
        f=fl[i,c];u=full['nativecache'][f['native_key']].numpy().astype(float);v=f['changed'].numpy().astype(float);ds['full']=center(v-u,w)
        e=energy(ds['0.01'],w);change=energy(ds['0.01']-ds['0.1'],w);relative=math.sqrt(change/e) if e else None
        converged=relative is not None and relative<=p['convergence_relative_norm_threshold']
        resolved=all(value is not None and value>=p['resolution_ratio_threshold'] for value in resolution.values())
        row=dict(target_index=i,context=c,kind=x['kind'],seed=x['seed'],family=x['family'],noop=x['noop'],local_energy=e,scale_difference_energy=change,scale_relative_difference=relative,resolution_ratios=resolution,scale_consistent=converged,resolved=resolved,usable=converged and resolved,full_vs_local=geometry(ds['0.01'],ds['full'],w))
        rows.append(row);responses[i,c]=ds
    by={(x['target_index'],x['context']):x for x in rows};pairs=[]
    for i in weights:
        for context in ['passage','sample']:
            a,b=by[i,'reference'],by[i,context]
            for amplitude in ['0.1','0.01','full']:
                pairs.append(dict(target_index=i,context=context,kind=b['kind'],seed=b['seed'],family=b['family'],amplitude=amplitude,jointly_usable=a['usable'] and b['usable'],**geometry(responses[i,'reference'][amplitude],responses[i,context][amplitude],weights[i])))
    summaries=[]
    for kind,seed in [('scalar',71),('scalar',72),('block',71),('block',72)]:
        for context in ['passage','sample']:
            for subset in ['all','jointly_usable']:
                for amplitude in ['0.1','0.01','full']:
                    rs=[x for x in pairs if x['kind']==kind and x['seed']==seed and x['context']==context and x['amplitude']==amplitude and (subset=='all' or x['jointly_usable'])]
                    total=sum(x['difference_energy'] for x in rs);ref=sum(x['reference_energy'] for x in rs);cos=[x['cosine'] for x in rs if x['cosine'] is not None]
                    summaries.append(dict(kind=kind,seed=seed,context=context,subset=subset,amplitude=amplitude,count=len(rs),nonzero=len(cos),relative_difference=ratio(total,ref),orthogonal_fraction=ratio(sum(x['orthogonal_energy'] for x in rs),total),median_cosine=statistics.median(cos) if cos else None))
    out=dict(inventory_sha256=sha(ip),protocol_sha256=sha(pp),rows=rows,pairs=pairs,summaries=summaries,counts=dict(rows=len(rows),noops=sum(x['noop'] for x in rows),scale_consistent=sum(x['scale_consistent'] for x in rows),resolved=sum(x['resolved'] for x in rows),usable=sum(x['usable'] for x in rows)),scope='Two-scale finite differences, not exact Jacobians. Resolution ratios only diagnose output storage precision. Familiar task data; no fitting or semantic-unit validation.')
    (P/'native_local_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['counts'])
    for s in summaries:
        if s['amplitude'] in ['0.01','full']:print(s)
if __name__=='__main__':main()

"""M110 independent quadratic formula census, finite limits, and causal metrics."""
import json,math
import numpy as np
import torch
from fit_pca_encoder import V,sha


def independent_crossings(a,b):
    # Explicit scalar quadratic formula, separate from polynomial root solver.
    C=(a*a).sum(1);B=2*(a*b).sum(1);A=(b*b).sum(1);roots=[]
    for i in range(32):
      for j in range(i):
        aa,bb,cc=A[i]-A[j],B[i]-B[j],C[i]-C[j]
        if aa==0:
          rr=[] if bb==0 else [-cc/bb]
        else:
          disc=bb*bb-4*aa*cc
          if disc<0:continue
          q=-.5*(bb+math.copysign(math.sqrt(disc),bb))
          rr=[q/aa,cc/q] if q else [-bb/(2*aa)]
        roots.extend(float(t) for t in rr if 0<t<1)
    roots.sort();knots=[0.]
    for t in roots:
        if t-knots[-1]>1e-10:knots.append(t)
    knots.append(1.)
    def top(t):return tuple(sorted(np.argsort(-((a+t*b)**2).sum(1),kind='stable')[:8].tolist()))
    intervals=[top((l+r)/2) for l,r in zip(knots[:-1],knots[1:])]
    return [(knots[i],intervals[i-1],intervals[i]) for i in range(1,len(knots)-1) if intervals[i-1]!=intervals[i]]


def main():
    torch.set_num_threads(1)
    rp=V/'gate_boundary_native.json';r=json.loads(rp.read_text());p=json.loads((V/'gate_boundary_protocol.json').read_text())
    assert r['source_sha256']==sha(V.parent/'probe_gate_boundaries.py') and r['protocol_sha256']==sha(V/'gate_boundary_protocol.json')
    assert sha(r['path'])==r['sha256'];saved=torch.load(r['path'],weights_only=True)
    files=json.loads((V/'block_change_roles_whole.json').read_text())['files'];files={f['pair_id']:f for f in files}
    states={}
    for e in p['models']:
      assert sha(e['path'])==e['sha256'];s=torch.load(e['path'],weights_only=True);states[(e['seed'],e['arm'])]=s
    samples={};rows=[];maxdecode=0.;maxroot=0.;totalboundaries=0;limitchecks=0
    for row in r['rows']:
      pid=row['pair_id'];key=f"{pid}|{row['seed']}|{row['arm']}"
      if pid not in samples:
        f=files[pid];assert sha(f['path'])==f['sha256'];d=torch.load(f['path'],weights_only=True);samples[pid]=[d['writes'][j][0,-1].double().numpy() for j in [0,1]]
      w0,w1=samples[pid];s=states[(row['seed'],row['arm'])];st={k:v.double().numpy() for k,v in s['state'].items()};mu=s['mean'].double().numpy();scale=s['scale'].double().numpy();D=st['dictionary'];D/=np.linalg.norm(D,axis=0)
      a=(((w0-mu)/scale)@st['encoder'].T+st['encoder_bias']).reshape(32,4);b=(((w1-w0)/scale)@st['encoder'].T).reshape(32,4)
      cc=independent_crossings(a,b);assert len(cc)==row['boundary_count'];totalboundaries+=len(cc)
      for (t,left,right),original in zip(cc,row['boundaries']):
        err=abs(t-original['t']);maxroot=max(maxroot,err);assert err<1e-8 and list(left)==original['left'] and list(right)==original['right']
      if not cc:assert row['status']=='no_crossing' and key not in saved;continue
      record=saved[key];assert np.allclose(record['a'].numpy(),a) and np.allclose(record['b'].numpy(),b)
      t,left,right=cc[0]
      def decode(t,selected):
        mask=np.zeros((32,4));mask[list(selected)]=1
        return ((a+t*b)*mask).reshape(-1)@D.T*scale+st['bias']*scale+mu
      lw=decode(t,left);rw=decode(t,right);nw=w0+t*(w1-w0)
      for mode,target in [('left_write',lw),('right_write',rw),('native_path_write',nw)]:
        err=float(np.linalg.norm(record[mode].numpy()-target)/max(np.linalg.norm(target),1));maxdecode=max(maxdecode,err);assert err<1e-9
      jump=rw-lw;assert np.isclose(np.linalg.norm(jump),row['jump_norm'])
      for f in row['finite_limits']:
        h=f['h'];diff=decode(t+h,right)-decode(t-h,left)
        for name,value in [('decoded_change_norm',np.linalg.norm(diff)),('native_change_norm',2*h*np.linalg.norm(w1-w0)),('limit_error_norm',np.linalg.norm(diff-jump))]:assert np.isclose(value,f[name],rtol=1e-7,atol=1e-8)
        limitchecks+=1
      out=record['outputs'];lp={k:x.double().log_softmax(-1).numpy() for k,x in out.items()}
      def kl(a,b):return float((np.exp(lp[a])*(lp[a]-lp[b])).sum())
      rows.append({k:row[k] for k in ['pair_id','family','seed','arm','first_boundary','jump_norm','jump_over_native_path']}|dict(KL_left_right=kl('left','right'),KL_right_left=kl('right','left'),KL_native_left=kl('native_path','left'),KL_native_right=kl('native_path','right'),top1_changed=int(out['left'].argmax()!=out['right'].argmax()),native_to_decoded_ratio_at_smallest_step=row['finite_limits'][-1]['decoded_change_norm']/max(row['finite_limits'][-1]['native_change_norm'],1e-30)))
    assert len(r['rows'])==224 and len({(x['pair_id'],x['seed'],x['arm']) for x in r['rows']})==224
    assert set(saved)=={f"{x['pair_id']}|{x['seed']}|{x['arm']}" for x in rows}
    assert r['forward_count']==56+3*len(rows) and r['baseline_replays']==56
    summaries=[]
    for family in ['ALL']+sorted({x['family'] for x in r['rows']}):
      for e in p['models']:
        rr=[x for x in rows if x['seed']==e['seed'] and x['arm']==e['arm'] and (family=='ALL' or x['family']==family)]
        total=sum(x['seed']==e['seed'] and x['arm']==e['arm'] and (family=='ALL' or x['family']==family) for x in r['rows'])
        summaries.append(dict(family=family,seed=e['seed'],arm=e['arm'],paths=total,crossing_paths=len(rr),top1_changes=sum(x['top1_changed'] for x in rr),mean_KL_left_right=float(np.mean([x['KL_left_right'] for x in rr])) if rr else None,median_jump_over_native_path=float(np.median([x['jump_over_native_path'] for x in rr])) if rr else None))
    result=dict(status='passed',source_sha256=sha(__file__),native_sha256=sha(rp),audit=dict(paths=224,total_boundaries=totalboundaries,limit_checks=limitchecks,max_root_discrepancy=maxroot,max_decode_relative_error=maxdecode),rows=rows,summaries=summaries)
    dest=V/'gate_boundary_analysis.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print(result['audit'])
    for s in summaries:
      if s['family']=='ALL':print(s)
    print('TOTAL_CROSSINGS',len(rows),'TOP1',sum(x['top1_changed'] for x in rows),'KL_RANGE',min(x['KL_left_right'] for x in rows),max(x['KL_left_right'] for x in rows))
if __name__=='__main__':main()

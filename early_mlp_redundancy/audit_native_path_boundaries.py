"""M114 independent derivative-interval root isolation and direct native limit checks."""
import json
import numpy as np
import torch
from scipy.optimize import brentq
from fit_pca_encoder import V,sha


def roots_by_brackets(c):
    c=np.trim_zeros(c,'b')
    if len(c)<=1:return []
    c=c/max(abs(c));critical=np.polynomial.polynomial.polyroots(np.arange(1,len(c))*c[1:]);knots=[0.]+sorted(float(z.real) for z in critical if abs(z.imag)<1e-9 and 0<z.real<1)+[1.]
    f=lambda t:float(sum(a*t**i for i,a in enumerate(c)))
    roots=[]
    for lo,hi in zip(knots[:-1],knots[1:]):
        if f(lo)*f(hi)<0:roots.append(brentq(f,lo,hi,xtol=1e-14,rtol=1e-14))
    # Tangencies cannot change support by themselves. Retain near-exact critical ties for interval accounting.
    roots.extend(t for t in knots[1:-1] if abs(f(t))<1e-12)
    return roots


def main():
    torch.set_num_threads(1);rp=V/'native_path_boundaries.json';r=json.loads(rp.read_text());assert r['source_sha256']==sha(V.parent/'measure_native_path_boundaries.py') and r['protocol_sha256']==sha(V/'native_path_boundary_protocol.json');assert sha(r['path'])==r['sha256'];saved=torch.load(r['path'],weights_only=True)
    receipt=json.loads((V/'matched_layer_operator_inputs.json').read_text());assert sha(receipt['path'])==receipt['sha256'];w=torch.load(receipt['path'],weights_only=True)['states'][1];L,R,D=[w[k].double().numpy() for k in ['L','R','D']]
    models={}
    for e in json.loads((V/'continuous_training_results.json').read_text())['models']:
        if e['mode']=='hard':assert sha(e['path'])==e['sha256'];models[e['seed']]=torch.load(e['path'],weights_only=True)
    maxroot=0.;maxmetric=0.;crossings=0;limits=0
    for row in r['rows']:
      c=saved[f"{row['seed']}|{row['pair_index']}"];n=c['code_numerators'].numpy();energy=np.zeros((32,5))
      for group in range(32):
       for atom in range(4):energy[group]+=np.convolve(n[:,group,atom],n[:,group,atom])
      roots=[]
      for i in range(32):
       for j in range(i):roots.extend(roots_by_brackets(energy[i]-energy[j]))
      roots.sort();knots=[0.]
      for t in roots:
       if t-knots[-1]>1e-9:knots.append(t)
      knots.append(1.)
      def support(t):
       raw=n[0]+t*n[1]+t*t*n[2];return tuple(sorted(np.argsort(-np.square(raw).sum(1),kind='stable')[:8].tolist()))
      ss=[support((lo+hi)/2) for lo,hi in zip(knots[:-1],knots[1:])];rr=[(knots[i],ss[i-1],ss[i]) for i in range(1,len(knots)-1) if ss[i-1]!=ss[i]]
      assert len(rr)==row['boundary_count'],(row['pair_index'],len(rr),row['boundary_count'])
      for (t,l,rr_),original in zip(rr,row['boundaries']):
       err=abs(t-original['t']);maxroot=max(maxroot,err);assert err<1e-7 and list(l)==original['left'] and list(rr_)==original['right'];crossings+=1
      if not rr:continue
      state=models[row['seed']];s={k:v.double().numpy() for k,v in state['state'].items()};mu=state['mean'].double().numpy();scale=state['scale'].double().numpy();B=s['dictionary']/np.linalg.norm(s['dictionary'],axis=0);z0=c['z0'].numpy();z1=c['z1'].numpy();t,left,right=rr[0]
      def direct(t):
       z=(1-t)*z0+t*z1;z=z/np.linalg.norm(z)*np.sqrt(len(z));write=D@((L@z)*(R@z));raw=(s['encoder']@((write-mu)/scale)+s['encoder_bias']).reshape(32,4);return write,raw
      def decode(raw,sel):
       q=np.zeros((32,4));q[list(sel)]=raw[list(sel)];return B@q.ravel()*scale
      def continuous(raw):
       norm=np.linalg.norm(raw,axis=1);threshold=np.sort(norm)[-9];code=raw*(np.maximum(norm-threshold,0)/np.where(norm>0,norm,1))[:,None];return B@code.ravel()*scale
      _,raw=direct(t);jump=decode(raw,right)-decode(raw,left);assert np.isclose(np.linalg.norm(jump),row['jump_norm'],rtol=1e-8,atol=1e-8)
      for record in row['finite_limits']:
       h=record['h'];wl,al=direct(t-h);wr,ar=direct(t+h);hard=decode(ar,right)-decode(al,left)
       vals=dict(native_change_norm=float(np.linalg.norm(wr-wl)),hard_change_norm=float(np.linalg.norm(hard)),continuous_change_norm=float(np.linalg.norm(continuous(ar)-continuous(al))),hard_limit_error=float(np.linalg.norm(hard-jump)))
       for key,value in vals.items():
        error=abs(value-record[key])/max(1,abs(value));maxmetric=max(maxmetric,error);assert error<1e-7,(key,error)
       limits+=1
    assert len(r['rows'])==32 and set(saved)=={f'{seed}|{i}' for seed in [71,72] for i in range(16)}
    out=dict(status='passed',source_sha256=sha(__file__),results_sha256=sha(rp),paths=32,crossings=crossings,finite_limit_cases=limits,max_root_difference=maxroot,max_scaled_metric_error=maxmetric,scope='Quartic root enumeration independently checked by derivative-interval Brent isolation; all first-crossing finite limits checked using direct native polynomial and independent NumPy shrinkage.')
    dest=V/'native_path_boundary_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()

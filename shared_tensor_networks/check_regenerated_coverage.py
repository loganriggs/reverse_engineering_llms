import itertools,json,tempfile
from pathlib import Path
import numpy as np
from regenerated_coverage import blocks,digest,save_readout_chunks,load_readout_chunks

root=Path('results/active_coverage');records=[]
for h,d,t in [(2,4,4),(4,8,14),(64,128,2084)]:
 rng=np.random.default_rng(255001+h);A=rng.normal(size=(h,d))/np.sqrt(d);B=rng.normal(size=(h,d))/np.sqrt(d);W=rng.normal(size=(t,h*(h+1)//2))/np.sqrt(h*(h+1)//2)
 n=17 if h<64 else 257
 first=list(blocks(A,B,W,255101,n));second=list(blocks(A,B,W,255101,n+5));replay=list(blocks(A,B,W,255101,n))
 for key in ['x','y','noise']:
  a=np.concatenate([b[key] for b in first]);b=np.concatenate([b[key] for b in second])[:n];c=np.concatenate([b[key] for b in replay])
  np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,c)
 for a,b in zip(first,replay):assert digest(a)==digest(b)
 # Independently form all unordered products with an explicit pair list.
 maxerr=0.
 for data in first:
  z=(data['x']@A.T)*(data['x']@B.T);phi=np.stack([z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(h),2)],1)
  ref=phi@W.T+.05*data['noise'];err=float(np.max(np.abs(ref-data['y'])));assert err<1e-10;maxerr=max(maxerr,err)
 # Distinct stream separates adaptation inputs from source, even at identical n.
 adapted=next(blocks(A,B,W,255101,n,stream=1));assert not np.array_equal(adapted['x'],first[0]['x'])
 with tempfile.TemporaryDirectory() as temp:
  chunks=save_readout_chunks(temp,'teacher',W);recovered=load_readout_chunks(temp,chunks);np.testing.assert_array_equal(W,recovered)
  sizes=[(Path(temp)/name).stat().st_size for name in chunks];assert max(sizes)<20*1024**2
  # Training final weights are float32; saving them as such loses no trained precision.
  final=W[:h*(h+1)//2].astype(np.float32);path=Path(temp)/'final.npz';np.savez_compressed(path,W=final)
  with np.load(path) as saved:np.testing.assert_array_equal(final,saved['W'])
  final_bytes=path.stat().st_size;assert final_bytes<20*1024**2
 records.append(dict(h=h,d=d,tasks=t,n=n,blocks=len(first),max_literal_error=maxerr,largest_teacher_chunk_bytes=max(sizes),final_readout_bytes=final_bytes,repeat_hashes=[digest(b) for b in first]))
(root/'regeneration_checks.json').write_text(json.dumps(dict(numpy_version=np.__version__,records=records,prefix_and_stream_checks=True),indent=2)+'\n');print(records)

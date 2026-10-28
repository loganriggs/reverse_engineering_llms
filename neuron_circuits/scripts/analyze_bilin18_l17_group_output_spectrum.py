"""H305 per-group output spectra and in-sample validation rank bounds."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_group_output_spectrum'
def main():
 cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());src=ART/'bilin18_l17_missing_output_spectrum_statistics.npz';assert hashlib.sha256(src.read_bytes()).hexdigest()==cfg['source_sha256'];stats=np.load(src);old=json.loads((ART/'bilin18_l17_missing_output_spectrum.json').read_text());out=[];bases=[]
 for i,(train,val) in enumerate(zip(stats['train'],stats['validation'])):
  et,U=np.linalg.eigh(train);U=U[:,::-1];et=et[::-1];ev=np.linalg.eigvalsh(val)[::-1];trace_t=np.trace(train);trace_v=np.trace(val);projected=np.einsum('ik,ij,jk->k',U,val,U,optimize=True);curves={}
  for k in cfg['ranks']:curves[str(k)]={'train':float(et[:k].sum()/trace_t),'validation':float(projected[:k].sum()/trace_v),'validation_oracle_bound':float(ev[:k].sum()/trace_v),'shared_validation':old['results']['equal_group'][str(k)]['validation']['groups'][i]}
  out.append({'group':i,'curves':curves,'train_rank80':int(np.searchsorted(np.cumsum(et)/trace_t,.8)+1),'validation_oracle_rank80':int(np.searchsorted(np.cumsum(ev)/trace_v,.8)+1),'trace_eigen_relative_error':float(abs(ev.sum()-trace_v)/trace_v)});bases.append(U[:,:256]);print('group',i,'rank80',out[-1]['validation_oracle_rank80'],flush=True)
 gates={'specialist32':all(v['curves']['32']['validation']>=.8 for v in out),'specialist128':all(v['curves']['128']['validation']>=.8 for v in out),'specialization_gain32':np.mean([v['curves']['32']['validation']-v['curves']['32']['shared_validation'] for v in out])>=.1,'oracle128':all(v['curves']['128']['validation_oracle_bound']>=.8 for v in out)};gates={k:bool(v) for k,v in gates.items()};np.savez_compressed(STEM.with_name(STEM.name+'_bases.npz'),group_bases=np.stack(bases));STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'groups':out,'gates':gates},indent=2)+'\n');print('GATES',gates,flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition'
def main():
    scaling=json.loads((ART/'bilin18_residual_scaling_documents.manifest.json').read_text());oldtest=json.loads((ART/'bilin18_distinct_document_panel.manifest.json').read_text());seen={r['document_token_sha256'] for r in scaling['records']+oldtest['records']};source=ROOT/scaling['source'];tokens=np.load(source,mmap_mode='r').reshape(-1);ends=np.flatnonzero(tokens==50256);starts=np.r_[0,ends[:-1]+1];eligible=np.flatnonzero((ends-starts>=129)&(ends<17000*513));rng=np.random.default_rng(20261120);samples=[];records=[]
    for i in rng.permutation(eligible):
        doc=tokens[starts[i]:ends[i]];digest=hashlib.sha256(doc.tobytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest);offset=int(rng.integers(len(doc)-129+1));sample=doc[offset:offset+129].copy();samples.append(sample);records.append({'relative_document_id':int(i),'document_token_sha256':digest,'absolute_token_start':int(starts[i]),'window_offset':offset,'window_token_sha256':hashlib.sha256(sample.tobytes()).hexdigest()})
        if len(samples)==128:break
    assert len(samples)==128 and len({r['window_token_sha256'] for r in records})==128
    path=ART/'bilin18_residual_replication_documents.npy';np.save(path,np.stack(samples));path.with_suffix('.manifest.json').write_text(json.dumps({'seed':20261120,'rows':128,'sequence_length':129,'source':scaling['source'],'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'New fixed test documents disjoint by full token hash from1024 discovery and original64 test docs. Complete cached docs ending before original row17000; older studies may have used text. No claim of global/training novelty.','records':records},indent=2));print('128 held-out documents built; excluded1024 discovery and64 old test docs')
if __name__=='__main__':main()

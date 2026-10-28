#!/usr/bin/env python3
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition'
def main():
    original=ART/'bilin18_depth_discovery_documents.npy';old=json.loads(original.with_suffix('.manifest.json').read_text());test=json.loads((ART/'bilin18_distinct_document_panel.manifest.json').read_text());source=ROOT/old['source'];tokens=np.load(source,mmap_mode='r').reshape(-1);ends=np.flatnonzero(tokens==50256);starts=np.r_[0,ends[:-1]+1];eligible=np.flatnonzero((ends-starts>=129)&(ends<17000*513));seen={r['document_token_sha256'] for r in old['records']+test['records']};samples=list(np.load(original));records=list(old['records']);rng=np.random.default_rng(20261119)
    for i in rng.permutation(eligible):
        doc=tokens[starts[i]:ends[i]];digest=hashlib.sha256(doc.tobytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest);offset=int(rng.integers(len(doc)-129+1));sample=doc[offset:offset+129].copy();samples.append(sample);records.append({'relative_document_id':int(i),'absolute_token_start':int(starts[i]),'document_length':len(doc),'window_offset':offset,'document_token_sha256':digest,'window_token_sha256':hashlib.sha256(sample.tobytes()).hexdigest()})
        if len(samples)==1024:break
    assert len(samples)==1024 and len({r['window_token_sha256'] for r in records})==1024
    array=np.stack(samples);assert np.array_equal(array[:64],np.load(original));path=ART/'bilin18_residual_scaling_documents.npy';np.save(path,array)
    manifest={'seed_for_additions':20261119,'rows':1024,'sequence_length':129,'source':old['source'],'source_sha256':old['source_sha256'],'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selection':'First64 exactly H244 discovery windows, then960 uniformly sampled eligible distinct complete cached documents ending before row17000, excluding original discovery and test document hashes; random129-token window per document.','scope':'Nested64/256/1024 discovery, disjoint from H243 test documents; cached text may appear in older studies. Compiled model and its calibration stay frozen.','records':records};path.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2));print('1024 distinct discovery docs; original64 prefix and test disjointness verified')
if __name__=='__main__':main()

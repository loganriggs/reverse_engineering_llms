#!/usr/bin/env python3
"""Fixed distinct-document panel from cached corpus; no claim of globally unused text."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition'
def main():
    source=ROOT/'artifacts/bilin18_layers_0_2/corpus/fineweb_viewer_20k.npy';tokens=np.load(source,mmap_mode='r').reshape(-1);ends=np.flatnonzero(tokens==50256);starts=np.r_[0,ends[:-1]+1]
    eligible=np.flatnonzero((ends-starts>=129)&(ends<17000*513));rng=np.random.default_rng(20261112);records=[];samples=[];test=json.loads((ART/'bilin18_distinct_document_panel.manifest.json').read_text());seen={v['document_token_sha256'] for v in test['records']}
    for i in rng.permutation(eligible):
        doc=tokens[starts[i]:ends[i]];digest=hashlib.sha256(doc.tobytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest);offset=int(rng.integers(len(doc)-129+1));sample=doc[offset:offset+129].copy()
        assert len(sample)==129 and not np.any(sample==50256)
        samples.append(sample);records.append({'relative_document_id':int(i),'absolute_token_start':int(starts[i]),'document_length':len(doc),'window_offset':offset,'document_token_sha256':digest,'window_token_sha256':hashlib.sha256(sample.tobytes()).hexdigest()})
        if len(samples)==64:break
    assert len(samples)==64 and len({x['window_token_sha256'] for x in records})==64
    output=ART/'bilin18_depth_discovery_documents.npy';np.save(output,np.stack(samples))
    manifest={'seed':20261112,'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'rows':64,'sequence_length':129,'selection':'Uniform sample without replacement of distinct eligible source documents ending before original row17000; length>=129, duplicate complete token documents excluded. Uniform random129-token window within each selected document. Model consumes first128 tokens.','scope':'Disjoint documents from H223 onward continuation panels. Cached text may have appeared in older ontology studies. No claim of globally unused text or new model-training holdout. Changes target population from token-weighted adjacent rows to document-weighted windows.','records':records}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Saved64 distinct documents; no duplicate windows; output SHA256',manifest['output_sha256'])
if __name__=='__main__':main()

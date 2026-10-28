"""H281 replication documents excluding all current discovery/test panels."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition'
def main():
    names=['bilin18_residual_scaling_documents','bilin18_distinct_document_panel','bilin18_residual_replication_documents','bilin18_local_chart_replication_documents','bilin18_feature_objective_documents','bilin18_core_interaction_replication_documents','bilin18_partner_rank_replication_documents','bilin18_cross_program_replication_documents','bilin18_cross_intact_replication_documents','bilin18_intercept_replication_documents','bilin18_kl_intercept_replication_documents','bilin18_kl_constant_audit_documents','bilin18_fixed_readout_replication_documents','bilin18_readout_learning_documents','bilin18_feature_comparison_documents','bilin18_feature_mixture_documents','bilin18_mixture_response_documents','bilin18_response_regularized_documents','bilin18_response_reversed_documents','bilin18_response_allheads_documents','bilin18_direction_confirmation_documents','bilin18_partner_factorial_confirmation_documents','bilin18_neighbor_confirmation_documents'];manifests=[json.loads((ART/f'{n}.manifest.json').read_text()) for n in names]
    seen={x['document_token_sha256'] for m in manifests for x in m['records']};excluded=set(seen);source=manifests[0]['source'];tokens=np.load(ROOT/source,mmap_mode='r').reshape(-1);ends=np.flatnonzero(tokens==50256);starts=np.r_[0,ends[:-1]+1];eligible=np.flatnonzero((ends-starts>=129)&(ends<17000*513));rng=np.random.default_rng(20270210);samples=[];records=[]
    for i in rng.permutation(eligible):
        doc=tokens[starts[i]:ends[i]];digest=hashlib.sha256(doc.tobytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest);offset=int(rng.integers(len(doc)-128));sample=doc[offset:offset+129].copy();samples.append(sample);records.append({'relative_document_id':int(i),'document_token_sha256':digest,'absolute_token_start':int(starts[i]),'window_offset':offset,'window_token_sha256':hashlib.sha256(sample.tobytes()).hexdigest()})
        if len(samples)==128:break
    assert len(samples)==128 and not ({x['document_token_sha256'] for x in records}&excluded)
    path=ART/'bilin18_kernel_confirmation_documents.npy';np.save(path,np.stack(samples));path.with_suffix('.manifest.json').write_text(json.dumps({'seed':20270210,'source':source,'rows':128,'sequence_length':129,'excluded_panels':names,'excluded_unique_documents':len(excluded),'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'New for this continuation, disjoint complete-document hashes from all listed panels; cached text may have appeared in older studies. Not model-training holdout.','records':records},indent=2));print('built128; excluded',len(excluded))
if __name__=='__main__':main()

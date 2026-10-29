"""M61 natural retention branch, separate from frozen M60 task loss."""
import json,torch
from balanced_response_loss import V,A,sha
from inspect_model import trace

def load_natural(e):
 path=A/'emlp_m36_natural_training_cache.pt';old=json.loads((V/'joint_shared_training_results.json').read_text());assert str(path)==old['natural_cache_path'] and sha(path)==old['natural_cache_sha256'];data=torch.load(path,weights_only=True);assert data['cohort_sha256']==sha(V/'natural_token_context_cohort.json') and data['checkpoint_sha256']==e.receipt.weights_sha256;assert set(data['natural'])==set(range(32));return data['natural'],path

def natural_loss(e,q,c):
 logits,_=trace.__wrapped__(e.model,c['ids'].cuda(),replace=lambda j,z,w:q+e.mlp.Down_bias if j==1 else w);e.calls+=1;lp=c['native'].cuda().double().log_softmax(-1);return (lp.exp()*(lp-logits.double().log_softmax(-1))).sum(-1).mean(),logits

"""M3: exact within-sample token/context variance partition; no held-out predictor claim."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch,tiktoken
P=Path(__file__).resolve().parent/'validation';meta_path=P/'empirical_spectra.json';meta=json.loads(meta_path.read_text());saved=torch.load('/tmp/emlp_m2_matrices.pt',weights_only=True);enc=tiktoken.get_encoding('gpt2')
tokens=np.array([meta['prefixes'][o['prefix_id']][-1] for o in meta['occurrences']]);unique_tokens=np.array([prefix[-1] for prefix in meta['prefixes']]);types=np.unique(tokens);contexts={int(t):int((unique_tokens==t).sum()) for t in types};rows=[]
def spectrum(x):
 s=np.linalg.svd(x,compute_uv=False);e=s*s;total=e.sum()
 if total<1e-20:return dict(energy=float(total),r90=0,r99=0,nuclear_effective_rank=0,singular_values=s.tolist())
 return dict(energy=float(total),r90=int(np.searchsorted(np.cumsum(e)/total,.9))+1,r99=int(np.searchsorted(np.cumsum(e)/total,.99))+1,nuclear_effective_rank=float(s.sum()**2/total),singular_values=s.tolist())
for sample in saved:
 Y=sample['Y'].double().numpy();layer=sample['layer']
 for sampling in ('occurrences','unique_prefixes'):
  indices=np.arange(len(Y)) if sampling=='occurrences' else np.array(meta['unique_indices']);tok=tokens[indices];y=Y[indices]
  for subset in ('all_tokens','repeated_context_tokens'):
   use=np.ones(len(y),dtype=bool) if subset=='all_tokens' else np.array([contexts[int(t)]>=2 for t in tok]);z=y[use];tt=tok[use];globalmean=z.mean(0);means=np.empty_like(z);groups=[]
   for t in np.unique(tt):
    select=tt==t;mu=z[select].mean(0);means[select]=mu;energy=float(np.square(z[select]-mu).sum())
    groups.append(dict(token=int(t),text=enc.decode([int(t)]),rows=int(select.sum()),distinct_prefixes=contexts[int(t)],within_energy=energy,mean_norm=float(np.linalg.norm(mu))))
   between=means-globalmean;within=z-means;total=float(np.square(z-globalmean).sum());be=float(np.square(between).sum());we=float(np.square(within).sum());raw=float(np.square(z).sum());cross=float(np.sum(between*within));assert abs(total-be-we)<1e-10*max(total,1)
   specs={}
   if subset=='all_tokens':specs=dict(between=spectrum(between),within=spectrum(within))
   groups.sort(key=lambda g:g['within_energy'],reverse=True)
   rows.append(dict(layer=layer,sampling=sampling,subset=subset,rows=len(z),token_types=len(groups),tokens_with_single_distinct_prefix=sum(contexts[g['token']]==1 for g in groups),rows_without_context_replication=sum(g['rows'] for g in groups if contexts[g['token']]==1),centered_energy=total,raw_energy=raw,between_token_energy=be,within_token_energy=we,between_fraction=be/total,within_fraction=we/total,within_over_raw_energy=we/raw,cross_inner_product=cross,mean_energy_fraction=float(len(z)*np.square(globalmean).sum()/raw),spectra=specs,groups=groups))
out=dict(status='completed_descriptive_partition',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),M2_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),model_forwards=0,unique_prefixes=len(unique_tokens),token_types=len(types),single_prefix_token_types=sum(v==1 for v in contexts.values()),rows=rows,protocol='Exact sample ANOVA of bias-excluded native writes: global mean + current-token mean deviation + within-token residual. Means and scoring on same opened panel; no unseen-context prediction or causal intervention. Repeated-context restriction requires >=2 distinct causal prefixes, not repeated identical occurrences. Within-token variation includes position/context/task covariation, not a uniquely isolated semantic correction.')
(P/'token_context_partition.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['unique_prefixes','token_types','single_prefix_token_types']},indent=2))
for r in rows:print(r['layer'],r['sampling'],r['subset'],r['rows'],round(r['between_fraction'],6),round(r['within_fraction'],6))

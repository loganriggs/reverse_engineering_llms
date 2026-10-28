"""H336 exploratory head/path stratification of already-observed moment panels."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_moment_head_heterogeneity'
def cosine(G,path='mlp'):
 if path=='mlp':dot=G[...,0,1]+G[...,0,2];e=G[...,1,1]+G[...,2,2]+2*G[...,1,2]
 else:k=1 if path=='compiled' else 2;dot=G[...,0,k];e=G[...,k,k]
 return dot/np.sqrt(G[...,0,0]*e)
def main():
 cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());rng=np.random.default_rng(cfg['seed']);boot=rng.integers(128,size=(4000,128));panels={};arrays={};max_replay=0.
 for panel,stem in cfg['panels'].items():
  path=ART/f'{stem}_rows.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['hashes'][panel];rows=np.load(path);old=json.loads((ART/f'{stem}.json').read_text());conditions=old['config']['conditions'];grams={name:np.stack([rows[f'g{i*9+h}_fisher_gram'] for h in range(9)]) for i,name in enumerate(conditions)};stats={}
  for name,G in grams.items():
   total=G.sum(1);stats[name]={'pooled_cosine':float(cosine(total.sum(0))),'mean_head_cosine':float(cosine(total).mean()),'heads':[]};max_replay=max(max_replay,abs(stats[name]['pooled_cosine']-old['summary'][name]['direct_mlp_cosine']))
   for h in range(9):stats[name]['heads'].append({'head':h,**{path:float(cosine(total[h],path)) for path in ['mlp','compiled','residual']},'direct_fisher_energy_share':float(total[h,0,0]/total[:,0,0].sum()),'mlp_fisher_energy_share':float((total[h,1,1]+total[h,2,2]+2*total[h,1,2])/(total[:,1,1]+total[:,2,2]+2*total[:,1,2]).sum())})
  contrasts={}
  for condition in ['centered','mean_cov','donor1']:
   contrasts[condition]={}
   for path in ['mlp','compiled','residual']:
    actual=grams['actual'];other=grams[condition];gap=cosine(other.sum(1),path)-cosine(actual.sum(1),path);draws=np.stack([cosine(other[h][boot].sum(1),path)-cosine(actual[h][boot].sum(1),path) for h in range(9)],1);radius=float(np.quantile(np.abs(draws-gap).max(1),.95));bands=np.stack([gap-radius,gap+radius],1);contrasts[condition][path]={'gaps':gap.tolist(),'simultaneous95':bands.tolist(),'max_deviation_radius':radius,'max_abs_gap':float(np.abs(gap).max()),'heads_excluding_zero':np.flatnonzero((bands[:,0]>0)|(bands[:,1]<0)).tolist()};arrays[f'{panel}_{condition}_{path}_bootstrap_gaps']=draws
  panels[panel]={'conditions':stats,'contrasts':contrasts}
 assert max_replay<1e-12
 d=np.array(panels['discovery']['contrasts']['mean_cov']['mlp']['gaps']);f=np.array(panels['confirmation']['contrasts']['mean_cov']['mlp']['gaps']);replication={'head_gap_correlation':float(np.corrcoef(d,f)[0,1]),'same_sign_count':int((np.sign(d)==np.sign(f)).sum())};fresh=panels['confirmation']['contrasts']['mean_cov']['mlp'];gates={'fresh_head_heterogeneity':fresh['max_abs_gap']>=.15 and len(fresh['heads_excluding_zero'])>0,'head_pattern_repeats':replication['head_gap_correlation']>=.7 and replication['same_sign_count']>=7};np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),**arrays);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'panels':panels,'replication':replication,'max_pooled_replay_error':max_replay,'gates':gates},indent=2)+'\n');print('FINISHED',fresh,replication,gates)
if __name__=='__main__':main()

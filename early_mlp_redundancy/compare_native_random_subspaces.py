"""M102 every random seed/rank versus fixed same-rank Euclidean PCA; exact controls."""
import json
from fit_pca_encoder import V,sha

def main():
 hashes={};rows=[];controls=0
 for cohort,tail in [('natural','natural'),('whole','factorial'),('source','source_factorial')]:
  results={};summaries={}
  for label,stem in [('data','native_mixed'),('random','native_random')]:
   rp=V/(stem+'_'+tail+'.json');ap=V/(stem+'_'+tail+'_audit.json');r=json.loads(rp.read_text());a=json.loads(ap.read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);hashes[rp.name]=sha(rp);hashes[ap.name]=sha(ap);results[label]=r
   if cohort=='natural':summaries[label]=r['summaries']
   else:
    cp=V/(stem+'_'+tail+'_comparison.json');c=json.loads(cp.read_text());assert c['result_sha256']==sha(rp);hashes[cp.name]=sha(cp);summaries[label]=c['summaries']
  key=(lambda x:(x['arm'],x['document'])) if cohort=='natural' else (lambda x:(x['arm'],x['key'],x['region'],x['corner']))
  d={key(x):{k:v for k,v in x.items() if k!='kind'} for x in results['data']['rows'] if x['arm'] in ['native','shared_linear']};n={key(x):{k:v for k,v in x.items() if k!='kind'} for x in results['random']['rows'] if x['arm'] in ['native','shared_linear']};assert d==n;controls+=len(d)
  ident=(lambda x:(x['arm'],'all')) if cohort=='natural' else (lambda x:(x['arm'],x['scope']))
  data={ident(x):x for x in summaries['data']}
  for b in summaries['random']:
   if b['arm'] in ['native','shared_linear']:continue
   arm,scope=ident(b);rank=int(arm.rsplit('__',1)[1]);seed=int(arm.split('__')[0].split('_')[1]);a=data['euclidean__'+str(rank),scope];fields=['mean_KL','CE','delta_CE','top1_agreement','local_relative_error'] if cohort=='natural' else [k for k in a if k.endswith(('_family','_pooled','_margin','_local')) or k in ['intact_KL','intact_binary_correct','intact_top1_correct','pair_both_correct']]
   rows.append(dict(cohort=cohort,seed=seed,rank=rank,scope=scope,metrics={k:dict(data_PCA=a[k],random=b[k],random_minus_data_PCA=b[k]-a[k]) for k in fields if a[k] is not None and b[k] is not None}))
 out=dict(status='completed',source_sha256=sha(__file__),inputs=hashes,unchanged_control_rows_exact=controls,rows=rows,scope='All3randomseeds×3ranks×allscope pairedagainstM101EuclideanPCA, samefunction/rank/mean/nativeweights. Notvariance/energymatched; orientationsarbitrary,no seedselection. Native/sharedlinear exactmetricreplay.')
 f=V/'native_random_subspace_comparison.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print('COMPARED',len(rows),'CONTROL_ROWS',controls)
if __name__=='__main__':main()

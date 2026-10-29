"""M101 all matched M100/M101 contrasts and unchanged-control replay."""
import json
from fit_pca_encoder import V,sha

def main():
 hashes={};rows=[];control_checks=0
 for cohort,tail in [('natural','natural'),('whole','factorial'),('source','source_factorial')]:
  results={};summaries={}
  for label,stem in [('restricted','native_operator'),('mixed','native_mixed')]:
   rp=V/(stem+'_'+tail+'.json');ap=V/(stem+'_'+tail+'_audit.json');r=json.loads(rp.read_text());a=json.loads(ap.read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);hashes[rp.name]=sha(rp);hashes[ap.name]=sha(ap);results[label]=r
   if cohort=='natural':summaries[label]=r['summaries']
   else:
    cp=V/(stem+'_'+tail+'_comparison.json');c=json.loads(cp.read_text());assert c['result_sha256']==sha(rp);hashes[cp.name]=sha(cp);summaries[label]=c['summaries']
  key=(lambda x:(x['arm'],x['document'])) if cohort=='natural' else (lambda x:(x['arm'],x['key'],x['region'],x['corner']))
  old={key(x):{k:v for k,v in x.items() if k!='kind'} for x in results['restricted']['rows'] if x['arm'] in ['native','shared_linear']};new={key(x):{k:v for k,v in x.items() if k!='kind'} for x in results['mixed']['rows'] if x['arm'] in ['native','shared_linear']};assert old==new;control_checks+=len(old)
  identity=(lambda x:(x['arm'],'all')) if cohort=='natural' else (lambda x:(x['arm'],x['scope']))
  olds={identity(x):x for x in summaries['restricted']};news={identity(x):x for x in summaries['mixed']};assert set(olds)==set(news)
  for ident,b in news.items():
   a=olds[ident];fields=['mean_KL','CE','delta_CE','top1_agreement','local_relative_error'] if cohort=='natural' else [k for k in a if k.endswith(('_family','_pooled','_margin','_local')) or k in ['intact_KL','intact_binary_correct','intact_top1_correct','pair_both_correct']]
   rows.append(dict(cohort=cohort,arm=ident[0],scope=ident[1],metrics={k:dict(restricted=a[k],mixed=b[k],mixed_minus_restricted=b[k]-a[k]) for k in fields if a[k] is not None and b[k] is not None}))
 out=dict(status='completed',source_sha256=sha(__file__),inputs=hashes,unchanged_control_rows_exact=control_checks,rows=rows,scope='Everybasis/rank/scope matched; native/sharedlinearcontrols exactmetric replay. Moremixedinteractioncapacity, notparametercountmatched. Noselection. Lowerloss/errorbetter,higheraccuracybetter; noaggregatepercentageunderstood.')
 f=V/'native_mixed_interaction_comparison.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print('COMPARED',len(rows),'CONTROL_ROWS',control_checks)
if __name__=='__main__':main()

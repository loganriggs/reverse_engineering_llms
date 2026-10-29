"""M74 fixed paired comparisons: local metric versus finite replacement."""
import json
import numpy as np
from downstream_metric_helpers import P,V,sha

def main():
    rp=V/'downstream_metric_results.json';r=json.loads(rp.read_text());audit=json.loads((V/'downstream_metric_audit.json').read_text())
    assert audit['status']=='passed' and audit['result_sha256']==sha(rp)
    execution=json.loads((V/'downstream_metric_execution.json').read_text());models=execution['models']
    summary=[]
    for e in models:
        rows=[row for row in r['rows'] if all(row[k]==e[k] for k in ['kind','seed','arm'])]
        assert len(rows)==32 and {x['document'] for x in rows}==set(range(32,64))
        s={k:e[k] for k in ['kind','seed','arm']}
        for key in ['Fisher_energy','residual_energy','relative_residual_energy']:
            s[key]=float(np.mean([x[key] for x in rows]))
        finite=[]
        for i,alpha in enumerate(execution['amplitudes']):
            kl=float(np.mean([x['finite'][i]['KL'] for x in rows]));pred=float(np.mean([x['finite'][i]['Taylor_KL'] for x in rows]))
            finite.append(dict(alpha=alpha,KL=kl,Taylor_KL=pred,actual_over_Taylor=kl/pred,mean_document_absolute_error=float(np.mean([x['finite'][i]['Taylor_absolute_error'] for x in rows])),log_energy_KL_correlation=float(np.corrcoef(np.log([x['Fisher_energy'] for x in rows]),np.log([x['finite'][i]['KL'] for x in rows]))[0,1])))
        s['finite']=finite;summary.append(s)
    pca=summary[0];assert pca['kind']=='pca';pairs=[];document_pairs=[]
    for s in summary[1:]:
        pair={k:s[k] for k in ['kind','seed','arm']}
        pair.update(PCA_lower_local_error=bool(pca['relative_residual_energy']<s['relative_residual_energy']),PCA_lower_Fisher=bool(pca['Fisher_energy']<s['Fisher_energy']),PCA_over_dense_Fisher=pca['Fisher_energy']/s['Fisher_energy'])
        pair['finite']=[dict(alpha=a['alpha'],PCA_lower_KL=bool(a['KL']<b['KL']),PCA_over_dense_KL=a['KL']/b['KL'],Fisher_ranking_matches=bool((pca['Fisher_energy']<s['Fisher_energy'])==(a['KL']<b['KL']))) for a,b in zip(pca['finite'],s['finite'])];pairs.append(pair)
        for doc in range(32,64):
            x=next(x for x in r['rows'] if x['kind']=='pca' and x['document']==doc)
            y=next(x for x in r['rows'] if x['document']==doc and all(x[k]==s[k] for k in ['kind','seed','arm']))
            document_pairs.append({k:s[k] for k in ['kind','seed','arm']}|dict(document=doc,PCA_lower_Fisher=bool(x['Fisher_energy']<y['Fisher_energy']),finite=[dict(alpha=a['alpha'],PCA_lower_KL=bool(a['KL']<b['KL']),Fisher_ranking_matches=bool((x['Fisher_energy']<y['Fisher_energy'])==(a['KL']<b['KL']))) for a,b in zip(x['finite'],y['finite'])]))
    counts=dict(PCA_lower_local=sum(x['PCA_lower_local_error'] for x in pairs),PCA_lower_Fisher=sum(x['PCA_lower_Fisher'] for x in pairs),aggregate_pairs=18,document_pairs=576,finite=[dict(alpha=alpha,PCA_lower_KL=sum(x['finite'][i]['PCA_lower_KL'] for x in pairs),aggregate_Fisher_ranking_matches=sum(x['finite'][i]['Fisher_ranking_matches'] for x in pairs),document_Fisher_ranking_matches=sum(x['finite'][i]['Fisher_ranking_matches'] for x in document_pairs)) for i,alpha in enumerate(execution['amplitudes'])])
    result=dict(status='completed',result_sha256=sha(rp),audit_sha256=sha(V/'downstream_metric_audit.json'),source_sha256=sha(__file__),summaries=summary,pairs=pairs,document_pairs=document_pairs,counts=counts,scope='Descriptive comparisons of fixed19representations on the same32natural documents. Shared data and training ancestry mean18comparisons are not independent replications. Taylor discrepancies include downstream map and output-distribution nonlinearity; no causality claim about training objective.')
    op=V/'downstream_metric_comparison.json';assert not op.exists();op.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(counts,indent=2));print('PCA',json.dumps(pca));print('DENSE_FISHER_RANGE',min(s['Fisher_energy'] for s in summary[1:]),max(s['Fisher_energy'] for s in summary[1:]));print('DENSE_RATIOS',[(alpha,min(s['finite'][i]['actual_over_Taylor'] for s in summary[1:]),max(s['finite'][i]['actual_over_Taylor'] for s in summary[1:])) for i,alpha in enumerate(execution['amplitudes'])])
if __name__=='__main__':main()

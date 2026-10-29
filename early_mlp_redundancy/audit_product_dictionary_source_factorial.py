"""M95 independent native-product predictor, delta and full-response verification for every case."""
import json
from collections import defaultdict
import numpy as np
import torch
from fit_pca_encoder import sha,V
from audit_pca_factorial import lp,margin
from independent_dictionary_predictions import IndependentDictionaryPrediction
P=V.parent

def main():
    torch.set_num_threads(1);rp=V/'product_dictionary_source_factorial.json';r=json.loads(rp.read_text());mp=V/'product_dictionary_evaluation_models.json'
    assert r['source_sha256']==sha(P/'probe_product_dictionary_source_factorial.py') and r['protocol_sha256']==sha(V/'product_dictionary_evaluation_execution.json') and r['models_sha256']==sha(mp)
    models=json.loads(mp.read_text())['models'];native=json.loads((V/'source_holdout_native.json').read_text());assert sha(native['samples_path'])==native['samples_sha256'];saved=torch.load(native['samples_path'],weights_only=True)
    nrows={x['sample_key']:x for x in native['records']};protocol=json.loads((V/'source_holdout_native_protocol.json').read_text());pairs={p['id']:p for p in protocol['pairs']}
    for pair in pairs.values():pair.setdefault('format','historical_source_group')
    key=lambda x:tuple(x[k] for k in ['kind','seed','arm','key','region'])
    rows={key(x)+(x['corner'],):x for x in r['rows']};geo={key(x):x for x in r['geometry']};invalid={key(x):x for x in r['ineligible']}
    assert len(rows)==1512 and len(geo)==504 and len(invalid)==0
    predictors={tuple(e[k] for k in ['kind','seed','arm']):IndependentDictionaryPrediction(e) for e in models}
    maximum=0.;maxdecode=0.;min_gap=float('inf');gatechecks=0;decodechecks=0;checked=0;geomchecks=0;ineligiblechecks=0;calls=28+sum(x['status']=='evaluated' and not x['noop'] for x in native['records'])+168
    def check(a,b):
        nonlocal maximum
        assert np.isclose(a,b,rtol=1e-8,atol=1e-9),(a,b);maximum=max(maximum,abs(a-b))
    for fi,file in enumerate(r['files']):
        assert sha(file['path'])==file['sha256'];s=torch.load(file['path'],weights_only=True,mmap=True);pid=s['pair_id'];assert pid==file['pair_id'];pair=pairs[pid];products=[next(iter(predictors.values())).prepare(z) for z in s['inputs']]
        for e in models:
            mk=tuple(e[k] for k in ['kind','seed','arm']);ms=':'.join(map(str,mk));recs=[]
            for ep in (0,1):
                w=s['writes'][ep].double().numpy();rec=s['reconstructions'][ms+f'|{ep}'];out=rec['write'].numpy();code=rec['code'].double().numpy();support=rec['support'].numpy()
                enc=predictors[mk];err,code_err=enc.verify(products[ep],rec['write'],rec['support'],rec['code']);maxdecode=max(maxdecode,err);decodechecks+=1;gatechecks+=int(e['variant']=='input');recs.append(out)
            for ep in (0,1):
                endpoint=pid+f'|{ep}';b=saved['baselines'][endpoint];base=b['logits'].numpy();native_lp=lp(base);prob=np.exp(native_lp);rb=s['baselines'][ms+f'|{ep}'].numpy()
                for region in ['changed_input_positions','final_position','identity']:
                    sk=endpoint+'|1|'+region;target=nrows[sk];kk=mk+(endpoint,region)
                    if target['status']!='evaluated':
                        row=invalid[kk];assert row['reason']==target['reason'] and row['format']==pair['format'];assert kk not in geo and all(kk+(c,) not in rows for c in ['reconstructed_native','native_reconstructed','reconstructed_reconstructed']);ineligiblechecks+=1;continue
                    pos=target['recipient_positions'];dp=target['donor_positions'];nd=np.zeros_like(s['writes'][ep].numpy());rd=np.zeros_like(nd);nd[:,pos]=s['writes'][1-ep].numpy()[:,dp]-s['writes'][ep].numpy()[:,pos];rd[:,pos]=recs[1-ep][:,dp]-recs[ep][:,pos];assert np.array_equal(nd[:,pos],saved['deltas'][sk]['delta'].numpy());gr=geo[kk];check(float(np.square(nd.astype(float)).sum()),gr['target_delta_energy']);check(float(np.square(rd.astype(float)-nd.astype(float)).sum()),gr['delta_error']);check(float(np.square(rd.astype(float)).sum()),gr['predicted_delta_energy']);geomchecks+=1
                    teacher=lp(saved['changed'][sk])-native_lp
                    for corner,delta,bg in [('reconstructed_native',nd,rb),('native_reconstructed',rd,base),('reconstructed_reconstructed',rd,rb)]:
                        z=s['changed'][ms+f'|{ep}|'+region+'|'+corner].numpy();row=rows[kk+(corner,)];assert row['format']==pair['format'];check(float((prob*((lp(z)-lp(bg))-teacher)**2).sum()),row['response_error']);effect=margin(z,b['groups'],b['variant'])-margin(bg,b['groups'],b['variant']);check((effect-target['margin_change'])**2,row['margin_error']);check(float((prob*teacher**2).sum()),row['target_energy']);check(target['margin_change']**2,row['margin_target_energy']);calls+=bool(np.count_nonzero(delta));checked+=1
                        if region=='identity':assert row['response_error']==row['margin_error']==gr['delta_error']==0
        print('AUDIT_PAIRS',fi+1,flush=True)
    grouped=defaultdict(list)
    for row in r['rows']:
        for family in [None,row['family']]:grouped[tuple(row[k] for k in ['kind','seed','arm','region','corner'])+(family,)].append(row)
    for summary in r['summaries']:
        rr=grouped[tuple(summary[k] for k in ['kind','seed','arm','region','corner','family'])];assert len(rr)==summary['n']
        for field,src in [('target_energy','target_energy'),('error_energy','response_error'),('margin_target_energy','margin_target_energy'),('margin_error_energy','margin_error')]:check(sum(x[src] for x in rr),summary[field])
        for numerator,denominator,field in [('error_energy','target_energy','relative_response_error'),('margin_error_energy','margin_target_energy','relative_margin_error')]:
            if summary[denominator]:check(summary[numerator]/summary[denominator],summary[field])
            else:assert summary[field] is None
    assert checked==1512 and geomchecks==504 and ineligiblechecks==0 and decodechecks==168 and gatechecks==168 and r['old_corner_replays']==84 and calls==r['forward_count']==1260
    result=dict(status='passed',result_sha256=sha(rp),source_sha256=sha(__file__),metric_helper_sha256=sha(P/'audit_pca_factorial.py'),response_vectors=checked,geometry_rows=geomchecks,ineligible_assignments=ineligiblechecks,decoder_checks=decodechecks,independent_gate_checks=gatechecks,prediction_helper_sha256=sha(P/'independent_dictionary_predictions.py'),max_decoder_relative_error=maxdecode,max_metric_absolute_error=maximum,forward_count=calls,scope='All saved final responses/deltas and summaries independently rescored. All predictions independently computed with float64 NumPy products, normalization, coefficient and decoder matrices. All variants use no native write. Support counts 128 dense output coordinates; all4608nativeproducts retained, no learned gates. Producer exactly replays28native baselines and84teacher targets. No independent transformer implementation.')
    op=V/'product_dictionary_source_factorial_audit.json';assert not op.exists();op.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()

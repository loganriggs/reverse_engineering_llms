"""Verify complete prospective pair/model/site/group coverage, including exclusions."""
import json
from fit_pca_encoder import V,sha


def main():
    pp=V/'block_change_roles_protocol.json';p=json.loads(pp.read_text());rows=[]
    for cohort,stem,field in [('train','training_causal_coverage','training_pair_ids'),('source','source_holdout_native','source_group_pair_ids'),('whole','whole_task_native','whole_task_pair_ids')]:
        rp=V/f'block_change_roles_{cohort}.json';r=json.loads(rp.read_text());ap=V/f'block_change_roles_{cohort}_audit.json';a=json.loads(ap.read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);wanted=p[field];assert [f['pair_id'] for f in r['files']]==wanted and len(wanted)==len(set(wanted));native=json.loads((V/(stem+'.json')).read_text());targets=[x for x in native['records'] if x['layer']==1 and x['pair_id'] in set(wanted)];assert len(targets)==6*len(wanted)
        key=lambda x:(x['pair_id'],x['seed'],x['arm'],x['endpoint'],x['region'])
        expected=set();ineligible=set()
        for t in targets:
            for e in p['models']:
                k=(t['pair_id'],e['seed'],e['arm'],t['endpoint'],t['region'])
                (expected if t['status']=='evaluated' else ineligible).add(k)
        assert {key(x) for x in r['controls']}==expected and len(r['controls'])==len(expected)
        assert {key(x) for x in r['ineligible']}==ineligible and len(r['ineligible'])==len(ineligible)
        assert {key(x)+(x['grouping'],) for x in r['interactions']}=={k+(g,) for k in expected for g in ['trained','shuffled']}
        actual={key(x)+(x['grouping'],x['group']) for x in r['rows']};goal={k+(g,i) for k in expected for g in ['trained','shuffled'] for i in range(32)};assert actual==goal and len(actual)==len(r['rows'])
        rows.append(dict(cohort=cohort,pairs=len(wanted),expected_group_records=len(goal),expected_joint_records=len(expected),ineligible_model_sites=len(ineligible),result_sha256=sha(rp),audit_sha256=sha(ap)))
    out=dict(status='passed',source_sha256=sha(__file__),protocol_sha256=sha(pp),rows=rows,scope='Exact expected Cartesian coverage fromfrozenpairIDs, nativeeligibility andfourmodels, all32groups/twogroupings/twodirections/threesites. No silent group/pair removal; all ineligible model-sites retained.')
    f=V/'block_change_roles_coverage_audit.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()

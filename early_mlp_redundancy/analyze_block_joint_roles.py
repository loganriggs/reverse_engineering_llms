"""Post-result full-change cross-seed diagnostic; no fitting or selected groups."""
import json
import numpy as np
from fit_pca_encoder import V,sha


def main():
    rows=[];inputs={}
    for cohort in ['train','source','whole']:
        rp=V/f'block_change_roles_{cohort}.json';r=json.loads(rp.read_text());a=json.loads((V/f'block_change_roles_{cohort}_audit.json').read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);inputs[rp.name]=sha(rp)
        for arm in sorted({x['arm'] for x in r['controls']}):
            for region in ['changed_input_positions','final_position']:
                rr=[x for x in r['controls'] if x['arm']==arm and x['region']==region];families=sorted({x['family'] for x in rr});vectors={seed:np.array([np.mean([x['margin_change'] for x in rr if x['seed']==seed and x['family']==f]) for f in families]) for seed in [71,72]};a=vectors[71];b=vectors[72];en=float(b@b);norm=float(np.linalg.norm(a)*np.linalg.norm(b));rows.append(dict(cohort=cohort,arm=arm,region=region,families=families,seed71_effects=a.tolist(),seed72_effects=b.tolist(),relative_error=float(np.square(a-b).sum()/en) if en else None,cosine=float(a@b/norm) if norm else None,target_energy=en))
    out=dict(status='completed',source_sha256=sha(__file__),inputs=inputs,rows=rows,scope='Explicit post-result diagnostic prompted by training/source-group weak role correspondence. Full encoded-change signed familymean effects across seeds, no group assignment/gain/fit. Allobjectives/cohorts/sites retained. Separates unstable block partition fromunstable wholeencoded task effects; neither outcome proves semantic units. Does not replace frozen group comparison.')
    dest=V/'block_change_joint_role_diagnostic.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('JOINT_DIAGNOSTIC',len(rows))
if __name__=='__main__':main()

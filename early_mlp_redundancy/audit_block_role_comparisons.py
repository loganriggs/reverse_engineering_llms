"""Scalar inverse-assignment verification of every fixed role prediction."""
import json,sys,math
import numpy as np
from fit_pca_encoder import V,sha


def main():
    cohort=sys.argv[1];rp=V/f'block_change_roles_{cohort}.json';cp=V/f'block_change_roles_{cohort}_comparison.json';r=json.loads(rp.read_text());c=json.loads(cp.read_text());assert c['result_sha256']==sha(rp);mp=V/'block_change_role_matching.json';assert c['matching_sha256']==sha(mp);matching=json.loads(mp.read_text());checks=0;maximum=0.
    def check(a,b):
        nonlocal checks,maximum
        if a is None or b is None:assert a is b;checks+=1;return
        assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b);maximum=max(maximum,abs(a-b));checks+=1
    signals={}
    for s in c['signatures']:
        for group in range(32):
            for fi,family in enumerate(s['families']):
                records=[x for x in r['rows'] if x['arm']==s['arm'] and x['grouping']==s['grouping'] and x['region']==s['region'] and x['seed']==s['seed'] and x['group']==group and x['family']==family]
                value=math.fsum(x['margin_change'] for x in records)/len(records);check(value,s['effects'][group][fi]);signals[s['arm'],s['grouping'],s['region'],s['seed'],group,family]=value
    for row in c['comparisons']:
        assignment=next(x for x in matching['rows'] if x['arm']==row['arm'] and x['grouping']==row['grouping'])
        def compute(permutation):
            inverse={target:source for source,target in enumerate(permutation)};assert len(inverse)==32;errors=[];energies=[];predenergy=0.;dot=0.
            for family in row['families']:
                err=0.;energy=0.
                for target in range(32):
                    a=signals[row['arm'],row['grouping'],row['region'],71,inverse[target],family];b=signals[row['arm'],row['grouping'],row['region'],72,target,family];err+=(a-b)**2;energy+=b*b;predenergy+=a*a;dot+=a*b
                errors.append(err);energies.append(energy)
            energy=sum(energies);error=sum(errors);norm=math.sqrt(predenergy*energy);familyerrors=[a/b if b else None for a,b in zip(errors,energies)];valid=[x for x in familyerrors if x is not None]
            return dict(relative_error=error/energy if energy else None,equal_family_error=sum(valid)/len(valid) if valid else None,cosine=dot/norm if norm else None,target_energy=energy,error_energy=error,family_errors=familyerrors,zero_target_families=sum(x is None for x in familyerrors))
        allscores=[]
        for perm,saved in [(assignment['target_group_for_source'],row['geometry'])]+list(zip(assignment['random_permutations'],row['random_metrics'])):
            expected=compute(perm);allscores.append(expected)
            for field in expected:
                if field=='family_errors':
                    for a,b in zip(expected[field],saved[field]):check(a,b)
                else:check(expected[field],saved[field])
        assert len(allscores)==101
        for field,summary in row['random_summary'].items():
            values=[x[field] for x in allscores[1:] if x[field] is not None]
            if not values:assert summary is None;continue
            expected=dict(mean=sum(values)/len(values),minimum=min(values),maximum=max(values),q05=float(np.quantile(values,.05)),q95=float(np.quantile(values,.95)))
            for k,v in expected.items():check(v,summary[k])
    for item in c['composition']:
        keys=['arm','seed','region'];cc=[x for x in r['controls'] if all(x[k]==item[k] for k in keys)];ii=[x for x in r['interactions'] if all(x[k]==item[k] for k in keys+['grouping'])];rr=[x for x in r['rows'] if all(x[k]==item[k] for k in keys+['grouping'])];den=sum(x['joint_response_energy'] for x in ii);teacher=sum(x['teacher_response_energy'] for x in cc)
        check(item['joint_response_energy'],den);check(item['nonadditivity_relative_to_joint'],sum(x['nonadditive_response_energy'] for x in ii)/den if den else None);check(item['encoded_teacher_relative_error'],sum(x['teacher_error'] for x in cc)/teacher if teacher else None);check(item['encoded_teacher_energy'],teacher);check(item['endpoints'],len(cc));check(item['group_noops'],sum(x['noop'] for x in rr));check(item['group_records'],len(rr))
    out=dict(status='passed',source_sha256=sha(__file__),comparison_sha256=sha(cp),result_sha256=sha(rp),matching_sha256=sha(mp),scalar_checks=checks,max_absolute_error=maximum,scope='All family/group signals rebuilt fromrawresponses. Everygeometric and100random predictions independently rescored usinginverse source-target assignment andscalar sums. All randomsummary andcomposition arithmetic checked. Fullvectoraudit remains separate.')
    dest=V/f'block_change_roles_{cohort}_comparison_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()

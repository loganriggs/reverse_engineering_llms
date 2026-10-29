"""Verify post-result joint-role diagnostic from raw joint controls."""
import json,math
from fit_pca_encoder import V,sha


def main():
    rp=V/'block_change_joint_role_diagnostic.json';r=json.loads(rp.read_text());checks=0;maximum=0.
    def check(a,b):
        nonlocal checks,maximum
        if a is None or b is None:assert a is b;checks+=1;return
        assert math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-10);maximum=max(maximum,abs(a-b));checks+=1
    for row in r['rows']:
        p=V/f"block_change_roles_{row['cohort']}.json";assert r['inputs'][p.name]==sha(p);raw=json.loads(p.read_text())['controls'];expected={}
        for seed in [71,72]:
            values=[]
            for family in row['families']:
                rr=[x for x in raw if x['seed']==seed and x['arm']==row['arm'] and x['region']==row['region'] and x['family']==family];values.append(math.fsum(x['margin_change'] for x in rr)/len(rr))
            for a,b in zip(values,row[f'seed{seed}_effects']):check(a,b)
            expected[seed]=values
        a,b=expected[71],expected[72];en=math.fsum(v*v for v in b);pn=math.fsum(v*v for v in a);err=math.fsum((x-y)**2 for x,y in zip(a,b));dot=math.fsum(x*y for x,y in zip(a,b));check(en,row['target_energy']);check(err/en if en else None,row['relative_error']);check(dot/math.sqrt(en*pn) if en*pn else None,row['cosine'])
    assert len(r['rows'])==12
    out=dict(status='passed',source_sha256=sha(__file__),result_sha256=sha(rp),checks=checks,max_absolute_error=maximum,scope='All12 post-result joint-role comparisons independently aggregated withscalar sums fromraw controls. Doesnot turn exploratory diagnostic into prospectivelyconfirmed evidence.')
    p=V/'block_change_joint_role_diagnostic_audit.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()

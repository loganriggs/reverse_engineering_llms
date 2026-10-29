"""M112 independent SQLite aggregate checks and explicit zero/eligibility accounting."""
import json,sqlite3,math
from fit_pca_encoder import V,sha


def main():
 path=V/'continuous_task_comparison.json';r=json.loads(path.read_text());conn=sqlite3.connect(':memory:');conn.execute('create table results(cohort, family, seed, region, condition, err real, target real, merr real, mtarget real, lerr real, ltarget real, candidate real)')
 for cohort in ['train','source','whole']:
  p=V/f'continuous_tasks_{cohort}.json';assert r['inputs'][p.name]==sha(p);data=json.loads(p.read_text())
  conn.executemany('insert into results values(?,?,?,?,?,?,?,?,?,?,?,?)',[(cohort,x['family'],x['seed'],x['region'],x['condition'],x['response_error'],x['teacher_response_energy'],x['margin_error'],x['teacher_margin_energy'],x['local_error'],x['native_delta_energy'],x['candidate_delta_energy']) for x in data['rows']])
  for x in data['rows']:
   if x['region']=='identity':assert all(x[k]==0 for k in ['response_error','teacher_response_energy','margin_error','teacher_margin_energy','local_error','native_delta_energy','candidate_delta_energy'])
 checks=0;maximum=0.
 for s in r['summaries']:
  where='cohort=? and seed=? and region=? and condition=?';args=[s['cohort'],s['seed'],s['region'],s['condition']]
  if s['view']=='qualified5':where+=" and family not in ('spatial_inverse','role_binding')"
  count=conn.execute('select count(*) from results where '+where,args).fetchone()[0];assert count==s['count']
  for label,num,den in [('response','err','target'),('margin','merr','mtarget'),('local','lerr','ltarget')]:
   pooled=conn.execute(f'select sum({num})/sum({den}) from results where '+where,args).fetchone()[0]
   equal=conn.execute(f'select avg(ratio) from (select sum({num})/sum({den}) as ratio from results where '+where+' group by family)',args).fetchone()[0]
   for k,val in [('pooled_'+label+'_error',pooled),('equal_family_'+label+'_error',equal)]:
    error=abs(val-s[k])/max(1,abs(val));maximum=max(maximum,error);assert error<1e-12;checks+=1
  ratio=conn.execute('select sum(candidate)/sum(ltarget) from results where '+where,args).fetchone()[0];assert math.isclose(ratio,s['candidate_over_native_energy'],rel_tol=1e-12,abs_tol=1e-12);checks+=1
 out=dict(status='passed',source_sha256=sha(__file__),comparison_sha256=sha(path),aggregate_checks=checks,max_scaled_discrepancy=maximum,zero_identity_rows_verified=True)
 dest=V/'continuous_task_comparison_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()

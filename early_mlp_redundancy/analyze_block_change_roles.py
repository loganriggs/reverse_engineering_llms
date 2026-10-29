"""Fixed geometry versus100 random assignments, with no fitted amplitudes."""
import json,sys
import numpy as np
from fit_pca_encoder import V,sha


def metrics(pred,target):
    energy=float(np.square(target).sum());error=float(np.square(pred-target).sum());norm=float(np.linalg.norm(pred)*np.linalg.norm(target));family=[]
    for j in range(target.shape[1]):
        en=float(np.square(target[:,j]).sum())
        family.append(float(np.square(pred[:,j]-target[:,j]).sum()/en) if en else None)
    valid=[x for x in family if x is not None]
    return dict(relative_error=error/energy if energy else None,equal_family_error=float(np.mean(valid)) if valid else None,cosine=float((pred*target).sum()/norm) if norm else None,target_energy=energy,error_energy=error,family_errors=family,zero_target_families=sum(x is None for x in family))


def main():
    cohort=sys.argv[1];assert cohort in ['train','source','whole'];rp=V/f'block_change_roles_{cohort}.json';ap=V/f'block_change_roles_{cohort}_audit.json';r=json.loads(rp.read_text());a=json.loads(ap.read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp)
    mp=V/'block_change_role_matching.json';matching=json.loads(mp.read_text());rows=[];signatures=[];composition=[]
    qualified={x['family'] for x in json.loads((V/'whole_task_native_comparison.json').read_text())['summaries'] if x['competent']} if cohort=='whole' else set()
    for assignment in matching['rows']:
        arm=assignment['arm'];grouping=assignment['grouping']
        for region in ['changed_input_positions','final_position']:
            raw=[x for x in r['rows'] if x['arm']==arm and x['grouping']==grouping and x['region']==region];families=sorted({x['family'] for x in raw});effects={}
            for seed in [71,72]:
                matrix=np.array([[np.mean([x['margin_change'] for x in raw if x['seed']==seed and x['group']==g and x['family']==fam]) for fam in families] for g in range(32)]);assert np.isfinite(matrix).all();effects[seed]=matrix;signatures.append(dict(arm=arm,grouping=grouping,region=region,seed=seed,families=families,effects=matrix.tolist()))
                cc=[x for x in r['controls'] if x['arm']==arm and x['seed']==seed and x['region']==region];ir=[x for x in r['interactions'] if x['arm']==arm and x['seed']==seed and x['region']==region and x['grouping']==grouping];den=sum(x['joint_response_energy'] for x in ir);teacher=sum(x['teacher_response_energy'] for x in cc)
                composition.append(dict(arm=arm,grouping=grouping,region=region,seed=seed,endpoints=len(cc),nonadditivity_relative_to_joint=sum(x['nonadditive_response_energy'] for x in ir)/den if den else None,joint_response_energy=den,encoded_teacher_relative_error=sum(x['teacher_error'] for x in cc)/teacher if teacher else None,encoded_teacher_energy=teacher,group_noops=sum(x['noop'] for x in raw if x['seed']==seed),group_records=sum(x['seed']==seed for x in raw)))
            scopes=[('all',list(range(len(families))))]+[('family:'+fam,[j]) for j,fam in enumerate(families)]
            if qualified:scopes.append(('contrast_qualified',[j for j,fam in enumerate(families) if fam in qualified]))
            for scope,indices in scopes:
                if not indices:continue
                source=effects[71][:,indices];target=effects[72][:,indices];pred=np.empty_like(source);pred[np.array(assignment['target_group_for_source'])]=source;geo=metrics(pred,target);null=[]
                for permutation in assignment['random_permutations']:
                    pred=np.empty_like(source);pred[np.array(permutation)]=source;null.append(metrics(pred,target))
                summary={}
                for field in ['relative_error','equal_family_error','cosine']:
                    values=[x[field] for x in null if x[field] is not None]
                    summary[field]=dict(mean=float(np.mean(values)),minimum=min(values),maximum=max(values),q05=float(np.quantile(values,.05)),q95=float(np.quantile(values,.95))) if values else None
                rows.append(dict(arm=arm,grouping=grouping,region=region,scope=scope,families=[families[j] for j in indices],geometry=geo,random_metrics=null,random_summary=summary))
    out=dict(status='completed',source_sha256=sha(__file__),result_sha256=sha(rp),audit_sha256=sha(ap),matching_sha256=sha(mp),cohort=cohort,signatures=signatures,comparisons=rows,composition=composition,scope='Signed family/site margin effects averaged across paired endpoints beforecrossseed prediction. Raw source71 effects copied togeometricallyassigned target72 groups; no amplitude fitting. Fixed100random permutations descriptive, not independent experimental replications/pvalues. Local group changes add; downstream response nonadditivity reported separately. Source/task holdouts are reused development evaluations.')
    dest=V/f'block_change_roles_{cohort}_comparison.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('ANALYZED',cohort,len(rows),'comparisons',len(signatures),'signatures')
if __name__=='__main__':main()

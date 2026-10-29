"""M76 all-model/family/format and contrast-qualified transfer comparison."""
import json
from collections import defaultdict
import numpy as np
import torch
from fit_pca_encoder import sha,V
from audit_pca_factorial import lp,margin
P=V.parent

def main():
    torch.set_num_threads(1);rp=V/'local_affine_chart_factorial.json';r=json.loads(rp.read_text());audit=json.loads((V/'local_affine_chart_factorial_audit.json').read_text());assert audit['status']=='passed' and audit['result_sha256']==sha(rp)
    models=json.loads((V/'local_affine_chart_evaluation_models.json').read_text())['models'];protocol=json.loads((V/'whole_task_native_protocol.json').read_text());pairs={p['id']:p for p in protocol['pairs']};competence=json.loads((V/'whole_task_native_comparison.json').read_text());qualified={s['family'] for s in competence['summaries'] if s['competent']}
    native=json.loads((V/'whole_task_native.json').read_text());assert sha(native['samples_path'])==native['samples_sha256'];saved=torch.load(native['samples_path'],weights_only=True)
    def ident(e):return tuple(e[k] for k in ['kind','seed','arm'])
    def scopes(family,fmt):return ['all','family:'+family,'format:'+family+':'+fmt]+(['contrast_qualified'] if family in qualified else [])
    intacts=[];ig=defaultdict(list)
    for file in r['files']:
        assert sha(file['path'])==file['sha256'];b=torch.load(file['path'],weights_only=True);p=pairs[file['pair_id']]
        for e in models:
            mk=ident(e);ms=':'.join(map(str,mk))
            for ep in (0,1):
                key=p['id']+f'|{ep}';n=saved['baselines'][key];z=b['baselines'][ms+f'|{ep}'];nl=lp(n['logits']);sl=lp(z);groups=n['groups'];variant=n['variant'];rec=b['reconstructions'][ms+f'|{ep}']
                row={k:e[k] for k in ['kind','seed','arm']}|dict(key=key,pair_id=p['id'],family=p['family'],format=p['format'],KL=float((np.exp(nl)*(nl-sl)).sum()),binary_correct=bool(margin(z,groups,variant)>0),top1_correct=bool(int(z.argmax()) in groups[variant]),native_binary_correct=bool(margin(n['logits'],groups,variant)>0),native_top1_correct=bool(int(n['logits'].argmax()) in groups[variant]),active_mean=float(rec['support'].double().sum(-1).mean()))
                intacts.append(row)
                for scope in scopes(p['family'],p['format']):ig[mk+(scope,)].append(row)
    rg=defaultdict(list);gg=defaultdict(list)
    for row in r['rows']:
        for scope in scopes(row['family'],row['format']):rg[ident(row)+(scope,row['region'],row['corner'])].append(row)
    for row in r['geometry']:
        for scope in scopes(row['family'],row['format']):gg[ident(row)+(scope,row['region'])].append(row)
    refs={}
    for name in ['local_affine_chart_natural.json']:
        for row in json.loads((V/name).read_text())['summaries']:refs[ident(row)]=row
    summaries=[]
    for key,rr in ig.items():
        mk=key[:3];scope=key[3];e=next(e for e in models if ident(e)==mk);s={k:e[k] for k in ['kind','seed','arm','variant']};s['base_arm']=e.get('base_arm',e['arm']);s.update(scope=scope,endpoints=len(rr),families=sorted({x['family'] for x in rr}))
        for field in ['KL','binary_correct','top1_correct','native_binary_correct','native_top1_correct','active_mean']:s['intact_'+field]=float(np.mean([x[field] for x in rr]))
        pp=defaultdict(list)
        for x in rr:pp[x['pair_id']].append(x)
        assert all(len(x)==2 for x in pp.values());s['intact_pair_both_correct']=float(np.mean([all(y['binary_correct'] for y in x) for x in pp.values()]));s['native_pair_both_correct']=float(np.mean([all(y['native_binary_correct'] for y in x) for x in pp.values()]))
        for region,site in [('changed_input_positions','source'),('final_position','final')]:
            geo=gg[mk+(scope,region)];en=sum(x['target_delta_energy'] for x in geo);s[site+'_local']=sum(x['delta_error'] for x in geo)/en if en else None
            for corner in ['reconstructed_native','native_reconstructed','reconstructed_reconstructed']:
                rows=rg[mk+(scope,region,corner)];te=sum(x['target_energy'] for x in rows);me=sum(x['margin_target_energy'] for x in rows);prefix=site+'_'+corner
                s[prefix+'_pooled']=sum(x['response_error'] for x in rows)/te if te else None;s[prefix+'_margin']=sum(x['margin_error'] for x in rows)/me if me else None;s[prefix+'_target_energy']=te;s[prefix+'_evaluated']=len(rows)
                per=[]
                for family in s['families']:
                    fr=[x for x in rows if x['family']==family];den=sum(x['target_energy'] for x in fr)
                    if den:per.append(sum(x['response_error'] for x in fr)/den)
                s[prefix+'_family']=float(np.mean(per)) if per else None;s[prefix+'_nonzero_families']=len(per)
        nr=refs[mk]
        for k in ['mean_KL','delta_CE','top1_agreement','local_relative_error','active_mean']:s['natural_'+k]=nr[k]
        summaries.append(s)
    reference_path=V/'whole_task_factorial_comparison.json';operator_path=V/'native_input_operator_comparison.json';protocol=json.loads((V/'local_affine_chart_protocol.json').read_text())
    for name,digest in protocol['references'].items():assert sha(V/name)==digest
    natural_audit=json.loads((V/'local_affine_chart_natural_audit.json').read_text());assert natural_audit['status']=='passed' and natural_audit['result_sha256']==sha(V/'local_affine_chart_natural.json')
    reference=json.loads(reference_path.read_text());comparisons=[]
    for left in summaries:
        for right in [x for x in reference['summaries'] if x['scope']==left['scope']]:
            for metric in [k for k in left if k.endswith(('_family','_pooled','_margin')) or k in ['intact_KL','intact_binary_correct','intact_top1_correct','source_local','final_local']]:
                a,b=left[metric],right[metric];higher=metric in ['intact_binary_correct','intact_top1_correct'];comparisons.append(dict(scope=left['scope'],reference_kind=right['kind'],reference_seed=right['seed'],reference_arm=right['arm'],reference_variant=right['variant'],metric=metric,affine=a,reference=b,affine_better=bool(a>b if higher else a<b) if a is not None and b is not None else None))
    operators=json.loads(operator_path.read_text());operator_comparisons=[]
    for left in summaries:
        for right in [x for x in operators['summaries'] if x['scope']==left['scope'] and x['region']!='identity']:
            site='source' if right['region']=='changed_input_positions' else 'final'
            operator_comparisons.append(dict(scope=left['scope'],region=right['region'],predictor=right['predictor'],affine_family_error=left[site+'_native_reconstructed_family'],operator_family_error=right['equal_family_response_error'],affine_local_error=left[site+'_local'],operator_local_error=right['local_relative_error']))
    result=dict(status='completed',source_sha256=sha(__file__),factorial_sha256=sha(rp),audit_sha256=sha(V/'local_affine_chart_factorial_audit.json'),natural_sha256=sha(V/'local_affine_chart_natural.json'),reference_sha256=sha(reference_path),operator_reference_sha256=sha(operator_path),summaries=summaries,intact_rows=intacts,comparisons=comparisons,operator_comparisons=operator_comparisons,scope='Three frozen local affine input-to-write predictors, fixed reference encoders and native operators. Not matched parameter count, latent dimension or computation. All sevenfamilies and14formats retained; historical evaluation exposure and training exclusions persist.')
    path=V/'local_affine_chart_comparison.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([x for x in summaries if x['scope'] in ['all','contrast_qualified']],indent=2))
if __name__=='__main__':main()

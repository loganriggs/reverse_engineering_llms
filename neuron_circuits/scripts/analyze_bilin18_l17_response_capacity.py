import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_response_capacity'
def main():
 j=json.loads(STEM.with_suffix('.json').read_text());fit=json.loads(STEM.with_name(STEM.name+'_selection.json').read_text());quad=fit['diagnostics']['quadratic1152'];tr=quad['fits']['0']['training_fidelity'];va=fit['validation']['quadratic1152_alpha0']['response_fidelity'];j['fit_diagnostics']=fit;j['gates']={'limited_empirical_capacity':tr<.5 and quad['fits']['0']['normal_relative_residual']<1e-7 and quad['rank']==quad['feature_count'],'generalization_gap':tr-va>=.15,'selected_response_transfer':all(j['results'][f'actual_s{s}_pooled']['quadratic1152_response']['residual_change_fidelity']>=.5 and j['results'][f'actual_s{s}_pooled']['quadratic1152_response']['residual_fidelity_document_bootstrap95'][0]>0 for s in [.1,1.])};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2));print('gates',j['gates'])
 for family in ['affine1152','quadratic1152']:
  d=fit['diagnostics'][family];print(family,'rank',d['rank'],'condition',d['retained_condition'])
  for role in ['unregularized','response']:
   name=fit['selected'][family+'_'+role];alpha=name.split('_alpha')[-1];print(role,name,'train',d['fits'][alpha]['training_fidelity'],'validation',fit['validation'][name],'test',j['results']['actual_s1.0_pooled'][family+'_'+role])
if __name__=='__main__':main()

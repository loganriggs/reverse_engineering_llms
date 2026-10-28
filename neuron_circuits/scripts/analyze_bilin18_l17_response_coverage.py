"""H263 preregistered paired comparisons of training-direction coverage."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_response_coverage_comparison'
def main():
 cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());base=np.load(ART/(cfg['reference']+'_rows.npz'));ix=np.random.default_rng(20261207).integers(128,size=(2000,128));contrasts={};audits={};name=cfg['primary_model']
 for variant in cfg['variants']:
  stem='bilin18_l17_response_supervision_'+variant;data=np.load(ART/(stem+'_rows.npz'));selected=json.loads((ART/(stem+'_selection.json')).read_text());audits[variant]={'normal_equation_residual_max':max(selected['normal_equation_relative_residual'].values()),'payload_norm_relative_max':max(v['norm_relative_max'] for v in selected['training_payload_audit'].values()),'payload_dot_global_relative_max':max(v['dot_global_relative'] for v in selected['training_payload_audit'].values())};contrasts[variant]={}
  for scale in [.1,1.]:
   for group,hs in [('supervised',[0,2,4,6,8]),('heldout',[1,3,5,7])]:
    def total(d,m):return sum(d[f'actual_s{scale}_h{h}_{m}'] for h in hs)
    energy=total(base,'residual_energy');e0=total(base,name+'_residual_error');e1=total(data,name+'_residual_error');k0=total(base,name+'_kl');k1=total(data,name+'_kl');de=e0-e1;dk=k0-k1
    contrasts[variant][f's{scale}_{group}']={'fidelity_gain_over_actual_training':float(de.sum()/energy.sum()),'fidelity_gain_bootstrap95':np.quantile(de[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist(),'conditional_kl_gain_over_actual_training':float(dk.sum()/k0.sum()),'kl_gain_bootstrap95':np.quantile(dk[ix].sum(1)/k0[ix].sum(1),[.025,.975]).tolist()}
 gates={label:all(contrasts[v][f's{s}_heldout']['fidelity_gain_over_actual_training']>=.1 and contrasts[v][f's{s}_heldout']['fidelity_gain_bootstrap95'][0]>0 for s in [.1,1.]) for v,label in [('angle','within_span_rescue'),('isotropic','ambient_coverage_rescue')]}
 STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'contrasts':contrasts,'gates':gates,'audits':audits},indent=2));print(json.dumps({'contrasts':contrasts,'gates':gates,'audits':audits},indent=2))
if __name__=='__main__':main()

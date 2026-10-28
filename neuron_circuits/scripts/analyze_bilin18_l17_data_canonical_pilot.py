from pathlib import Path
import numpy as np,json
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';s='bilin18_l17_data_canonical_pilot';j=json.loads((A/f'{s}.json').read_text());z=np.load(A/f'{s}_rows.npz');rng=np.random.default_rng(j['config']['bootstrap_seed']);ix=rng.integers(0,128,(4000,128));ci={}
for mode in ['intact','negative','positive']:
 for method in ['write','kl','ce']:
  v=z[mode+'_'+method];assert np.isfinite(v).all();ci[mode+'_'+method]=np.quantile(v[ix].mean(1),[.025,.975],axis=0).T.tolist()
# Check empirical input-coordinate canonical metric and equivalent latent parameterizations.
c=np.load(A/f'{s}_canonical_maps.npz');errors={};rot=np.linalg.qr(rng.standard_normal((16,16)))[0]
for method in ['write','kl','ce']:
 enc=c[method+'_encoder'];sv=c[method+'_singular_values'];dec=c[method+'_decoder'];base=(enc*sv)@dec;other=((enc*sv)@rot)@(rot.T@dec);errors[method]=float(np.linalg.norm(other-base)/np.linalg.norm(base));assert errors[method]<1e-12
j.update(descriptive_document_bootstrap95=ci,latent_gauge_rotation_relative_error=errors,comparison_interpretation='Both objective-fitted candidates selected step0. Equal final maps are a selection outcome, not evidence objectives have equivalent optima. No confirmatory gates.')
(A/f'{s}.json').write_text(json.dumps(j,indent=2)+'\n');print('Exploratory document intervals',ci['intact_write'],'gauge audit',errors)

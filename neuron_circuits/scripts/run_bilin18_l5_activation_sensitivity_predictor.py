"""H770 fixed small activation-only regressors for H769 sensitivity gain."""
import json,time
from pathlib import Path
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_activation_sensitivity_predictor'
def fit_predict(X,y,nt,ridge=.1):
 mean=X[:nt].mean(0);std=X[:nt].std(0,unbiased=False).clamp_min(1e-12);z=(X-mean)/std;ly=y.log();intercept=ly[:nt].mean();W=torch.linalg.solve(z[:nt].T@z[:nt]/nt+ridge*torch.eye(X.shape[1],dtype=torch.double),z[:nt].T@(ly[:nt]-intercept)/nt);logpred=intercept+z@W;raw=logpred.exp();calibration=y[:nt].mean()/raw[:nt].mean();return raw*calibration,{'mean':mean,'std':std,'coefficients':W,'log_intercept':intercept,'calibration':calibration}
def main():
 torch.set_num_threads(6);start=time.time();paths={'data':ART/'bilin18_l5_natural_featurizer_baselines_data.pt','base':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','probes':ART/'bilin18_l5_disjoint_sensitivity_probes_captures.pt','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};p=torch.load(paths['probes'],weights_only=False);cc=p['contexts'];nt=sum(c['split']=='train' for c in cc);cfg={'experiment':'H770','hashes':hashes,'contexts':cc,'ridge':.1,'features':['log_prefix_length','log_raw_head_norm','log_centered_head_norm','log_centered_write_norm','top8_write_coordinate_energy_fraction'],'models':['length_only','five_stats','five_stats_plus_eight_write_PCs'],'design':'Fit log scalar gain from first12 H769 probes on32 training documents. Fixed ridge.1 and no selection. Train-standardize activation inputs; exponentiate predictions and calibrate arithmetic mean on training only. Evaluate gain and12targetcurvatures on16 separate validation documents. Baseline PCA and centering fitted on original training data only.','limitations':'Small reused validation panel; no output/model probes needed for predictor inference, but labels came from probes. Target-probe prediction is not actual finite KL or causal fidelity. No learned featurizer evaluated.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['data'],weights_only=False);base=torch.load(paths['base'],weights_only=False)['fits']['L5H5'];raw=torch.stack([data['activations']['L5H5'][c['doc'],c['length']//8-1] for c in cc]).double();center=raw-base['mean'];coords=center@base['write_pca']['encode'];energy=coords.square();stats=torch.stack([torch.tensor([c['length'] for c in cc],dtype=torch.double).log(),raw.norm(dim=1).log(),center.norm(dim=1).log(),(center@base['write'].T).norm(dim=1).log(),energy.topk(8,dim=1).values.sum(1)/energy.sum(1)],1);features={'length_only':stats[:,:1],'five_stats':stats,'five_stats_plus_eight_write_PCs':torch.cat([stats,coords[:,:8]],1)};gain=p['gain'];predictions={'static':torch.ones_like(gain),'probe_scalar':gain};fits={}
 for name,X in features.items():predictions[name],fits[name]=fit_predict(X,gain,nt)
 # Label leakage check: changing validation labels cannot affect fitted prediction.
 altered=gain.clone();altered[nt:]*=1000;leak=max(float((fit_predict(X,altered,nt)[0]-predictions[name]).abs().max()) for name,X in features.items())
 summary=[];records=[];gen=torch.Generator().manual_seed(770100);draw=torch.randint(16,(10000,16),generator=gen);actual=p['curvatures'][:,12:];targetmean=p['means'][12:];static_error=(targetmean-actual[nt:]).square().mean(1).reshape(16,2).mean(1)
 for name,g in predictions.items():
  pred=g[:,None]*targetmean
  for split,sl in [('train',slice(0,nt)),('validation',slice(nt,None))]:
   a=actual[sl];q=pred[sl];truth=gain[sl];gp=g[sl];row={'method':name,'split':split,'gain_relative_RMS_error':float((gp-truth).norm()/truth.norm()),'gain_log_RMSE':float((gp.log()-truth.log()).square().mean().sqrt()),'target_curvature_relative_RMS_error':float((q-a).norm()/a.norm()),'target_curvature_median_relative_error':float(((q-a).abs()/a).median())}
   if split=='validation':
    err=(q-a).square().mean(1).reshape(16,2).mean(1);difference=(err-static_error)[draw].mean(1);row['paired_document_bootstrap95_MSE_minus_static']=[float(torch.quantile(difference,.025)),float(torch.quantile(difference,.975))]
   summary.append(row)
  for i,c in enumerate(cc):records.append({**c,'method':name,'gain':float(gain[i]),'predicted_gain':float(g[i]),'target_actual':actual[i].tolist(),'target_prediction':pred[i].tolist()})
 audits={'hashes_unchanged':all(sha(pth)==hashes[k] for k,pth in paths.items()),'finite_features_and_predictions':bool(torch.isfinite(stats).all() and all(torch.isfinite(g).all() for g in predictions.values())),'validation_label_mutation_prediction_error':leak,'train_documents':len({c['doc'] for c in cc[:nt]}),'validation_documents':len({c['doc'] for c in cc[nt:]}),'whole_model_forwards':0,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite_features_and_predictions'] and leak==0
 torch.save({'fits':fits,'features':features,'predictions':predictions,'contexts':cc},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'summary':summary,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits},indent=2))
if __name__=='__main__':main()

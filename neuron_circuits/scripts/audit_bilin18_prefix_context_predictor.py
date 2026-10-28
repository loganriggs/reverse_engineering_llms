"""Training-only optimizer refinement audit; frozen evaluation models unchanged."""
import torch,json
from run_bilin18_diverse_causal_tasks import ART

def main():
 torch.set_num_threads(6);f=torch.load(ART/'bilin18_prefix_context_predictor_locked_folds.pt',weights_only=False);d=torch.load(ART/'bilin18_metric_wording_edit_factorial_captures.pt',weights_only=False);out=[]
 for fam,fold in f['folds'].items():
  for which,keys in [('shared',fold['train']),('specific',fold['own'])]:
   p=fold[which];z=torch.stack([d['inputs'][k] for k in keys]).double();target=torch.stack([d['inputs'][k[0],k[1]+2] for k in keys]).double();u=p['shift'].clone().requires_grad_(True);opt=torch.optim.LBFGS([u],max_iter=200,tolerance_grad=1e-10,tolerance_change=1e-13,line_search_fn='strong_wolfe')
   def pred(u):
    v=z+u;return v/v.norm(dim=-1,keepdim=True)*1152**.5
   old=pred(u).detach();oldloss=float((old-target).square().mean())
   def closure():
    opt.zero_grad();loss=(pred(u)-target).square().mean();loss.backward();return loss
   opt.step(closure);loss=closure();new=pred(u).detach();out.append(dict(family=fam,fit=which,old_loss=oldloss,refined_loss=float(loss.detach()),relative_loss_reduction=(oldloss-float(loss.detach()))/oldloss,training_prediction_change_relative_to_edit=float((new-old).norm()/(target-z).norm()),refined_gradient_norm=float(u.grad.norm())))
 (ART/'bilin18_prefix_context_predictor_optimization_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()

"""Independent forward-map audit of H598's fixed response metric."""
import json,torch
from run_bilin18_diverse_causal_tasks import ART
from run_bilin18_qkov_causal_edges import load_full_model
@torch.inference_mode()
def main():
 torch.set_num_threads(6);m=torch.load(ART/'bilin18_number_fixed_response_metric_metric.pt',weights_only=False);old=torch.load(ART/'bilin18_number_mlp11_response_cross_captures.pt',weights_only=False);model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):mod.float().eval()
 mlp=model.blocks[11].mlp;A=m['basis'];K=m['K'];test=torch.randn(4,A.shape[1],generator=torch.Generator().manual_seed(598));coeff=test@K.T;pc=coeff@A;eu=(test@A.T)@A;rhs=torch.zeros(64,4,dtype=torch.float64);loss=torch.zeros(3,dtype=torch.float64)
 def T(c,e):
  c=c.cuda();e=e.cuda();return mlp.Down(mlp.Left(c)*mlp.Right(e)+mlp.Left(e)*mlp.Right(c)).cpu().double()
 for k in m['training_ids']:
  e=old['inputs'][k,.0625]-old['inputs'][k,0.];M=T(A,e).T;target=T(test,e).T;rhs+=M.T@target/len(m['training_ids']);p=T(pc,e).T;q=T(eu,e).T;loss+=torch.stack([(p-target).square().sum(),(q-target).square().sum(),target.square().sum()])/len(m['training_ids'])
 analytic=m['cross']@test.double().T;normal=m['H']@coeff.double().T-rhs;result=dict(training_responses=len(m['training_ids']),test_vectors=4,adjoint_formula_relative_error=float((rhs-analytic).norm()/rhs.norm()),normal_equation_forward_map_relative_error=float(normal.norm()/rhs.norm()),metric_to_euclidean_training_response_squared_error_ratio=float(loss[0]/loss[1]));assert result['adjoint_formula_relative_error']<2e-6 and result['normal_equation_forward_map_relative_error']<2e-6 and result['metric_to_euclidean_training_response_squared_error_ratio']<=1.00001;(ART/'bilin18_fixed_response_metric_audit.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()

"""M97 single failed case: scalar reduction versus native VJP precision.

Float64 is a numerical reference function, not a replay of float32 benchmark
outputs. Stored rotary constants are preserved; final logits stay float64 only
in the explicitly labeled reference mode.
"""
import json
import torch
from torch.nn import functional as F
from fit_pca_encoder import V,sha
from bilinear_input_causal_loss import InputCausalExperiment
from product_dictionary_direction import FrozenDictionaryDirection,scalar_derivative


def forward(model,ids,qr=None,delta=None,cast_output=True):
 x0=F.rms_norm(model.transformer.wte(ids),(model.config.n_embd,));x=x0;bus=None
 for j,block in enumerate(model.transformer.h):
  mixed=block.lambdas[0]*x+block.lambdas[1]*x0;attn,bus=block.attn(F.rms_norm(mixed,(mixed.shape[-1],)),bus);pre=mixed+attn;z=F.rms_norm(pre,(pre.shape[-1],));native=block.mlp(z)
  write=(native+delta if delta is not None else qr+block.mlp.Down_bias) if j==1 else native;x=pre+write
 logits=30*torch.tanh(model.lm_head(F.rms_norm(x,(x.shape[-1],)))/30)
 return logits.float()[0,-1] if cast_output else logits[0,-1]


def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=V/'product_dictionary_direction_precision_protocol.json';protocol=json.loads(pp.read_text())
 for name,digest in protocol['references'].items():assert sha(V/name)==digest
 e=InputCausalExperiment();direction=FrozenDictionaryDirection(72);pairs=json.loads((V/'source_holdout_native_protocol.json').read_text())['pairs'];pair=pairs[0];assert pair['id']=='old:capital_recall-context0-0';prior=json.loads((V/'product_dictionary_source_factorial.json').read_text());file=next(f for f in prior['files'] if f['pair_id']==pair['id']);assert sha(file['path'])==file['sha256'];blob=torch.load(file['path'],weights_only=True);nr=json.loads((V/'source_holdout_native.json').read_text());assert sha(nr['samples_path'])==nr['samples_sha256'];native=torch.load(nr['samples_path'],weights_only=True);endpoint=1;key=pair['id']+'|1';ids=torch.tensor([pair['token_ids'][endpoint]],device='cuda');prepared=[direction.prepare(z.cuda()) for z in blob['inputs']];base=native['baselines'][key]['logits'].cuda();nlp=base.double().log_softmax(-1);records=[r for r in nr['records'] if r['pair_id']==pair['id'] and r['endpoint']==endpoint and r['status']=='evaluated'];results=[];calls=0
 for mode in ['original_float32','double_tangent_reduction','double_native_reference']:
  full=mode=='double_native_reference';wide=mode!='original_float32'
  if full:e.model.double()
  t=torch.tensor(0.,device='cuda',dtype=torch.float64 if wide else torch.float32,requires_grad=True);writes=[];tangents=[]
  for ends,tangent in prepared:
   if wide:q=ends[0].double()+t*tangent.double();q=q if full else q.float()
   else:q=ends[0]+t*tangent
   writes.append(q);tangents.append(tangent.double())
  qr,qd=writes[endpoint],writes[1-endpoint];intact=forward(e.model,ids,qr=qr,cast_output=not full);calls+=1;K=(nlp.exp()*(nlp-intact.double().log_softmax(-1))).sum();branches=[];balanced=[]
  for target in records:
   region=target['region'];pos,dp=target['recipient_positions'],target['donor_positions'];nd=torch.zeros_like(qr);nd[:,pos]=native['deltas'][target['sample_key']]['delta'].to(qr);rd=torch.zeros_like(qr)
   if pos:rd=rd.index_copy(1,torch.tensor(pos,device='cuda'),qd[:,dp]-qr[:,pos])
   teacher=native['changed'][target['sample_key']].cuda().double().log_softmax(-1)-nlp
   for corner,delta,bg in [('native_reconstructed',rd,None),('reconstructed_native',nd,qr),('reconstructed_reconstructed',rd,qr)]:
    baseline=base if corner=='native_reconstructed' else intact
    if region=='identity':changed=baseline
    else:changed=forward(e.model,ids,delta=delta,cast_output=not full) if corner=='native_reconstructed' else forward(e.model,ids,qr=bg+delta,cast_output=not full);calls+=1
    R=(nlp.exp()*((changed.double().log_softmax(-1)-baseline.double().log_softmax(-1))-teacher).square()).sum();branches.append((region,corner,R,changed,baseline))
    if region!='identity':balanced.append(R/e.den[pair['family'],region])
  B=torch.stack(balanced).mean();direct=scalar_derivative(B,t);gB=torch.autograd.grad(B,(qr,qd),retain_graph=True);sum_grad=[torch.zeros_like(qr,dtype=torch.float64),torch.zeros_like(qd,dtype=torch.float64)];terms=[]
  for region,corner,R,changed,baseline in branches:
   slope=scalar_derivative(R,t)
   if not full:
    saved=blob['changed'][f'product_dictionary:72:intact_KL|1|{region}|{corner}'];assert torch.equal(changed.detach().cpu(),saved)
   if region=='identity':assert float(R.detach())==slope==0;continue
   weight=1/(e.den[pair['family'],region]*len(balanced));terms.append(slope*weight);grads=torch.autograd.grad(R,(qr,qd),retain_graph=True,allow_unused=True)
   for j,g in enumerate(grads):
    if g is not None:sum_grad[j]+=weight*g.double()
  weighted=sum(terms);tangent_pair=[tangents[endpoint],tangents[1-endpoint]];direct_dot=float(sum((g.double()*v).sum() for g,v in zip(gB,tangent_pair)));branch_dot=float(sum((g*v).sum() for g,v in zip(sum_grad,tangent_pair)));graderror=float(torch.sqrt(sum((g.double()-s).square().sum() for g,s in zip(gB,sum_grad)))/torch.sqrt(sum(s.square().sum() for s in sum_grad)));tol=1e-7+1e-5*abs(direct)
  if not full:assert torch.equal(intact.detach().cpu(),blob['baselines']['product_dictionary:72:intact_KL|1'])
  record=dict(mode=mode,K=float(K.detach()),B=float(B.detach()),direct_B_slope=direct,weighted_branch_slope_sum=weighted,absolute_discrepancy=abs(direct-weighted),original_tolerance=tol,passes_original_tolerance=abs(direct-weighted)<=tol,direct_write_gradient_double_dot=direct_dot,weighted_branch_write_gradients_double_dot=branch_dot,write_gradient_accounting_relative_error=graderror,max_intact_logit_difference_from_float32=float((intact.detach().double().cpu()-blob['baselines']['product_dictionary:72:intact_KL|1'].double()).abs().max()),branch_contributions=terms)
  results.append(record);print(json.dumps(record),flush=True);del qr,qd,writes,K,B,branches,balanced,gB,sum_grad
 out=dict(status='diagnostic_completed',source_sha256=sha(__file__),protocol_sha256=sha(pp),case=dict(seed=72,key=key,anchor=0),rows=results,forward_count=calls,scope='Same failed case, three declared precision modes. Firsttwo replay float32 endpoints exactly; double native reference changes arithmetic and default RMS epsilon and is not a benchmark substitution. No tolerance update or global derivative validity claim.')
 path=V/'product_dictionary_direction_precision.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls,flush=True)
if __name__=='__main__':main()

import json,math
from collections import defaultdict
import torch
from run_bilin18_diverse_causal_tasks import ART,measure
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
@torch.inference_mode()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');s='bilin18_size_controlled_bilinear_transfer';j=json.loads((ART/f'{s}.json').read_text());c=torch.load(ART/f'{s}_captures.pt',weights_only=False);examples=c['examples'];rows=[r for r in j['records'] if r['rho']==.3 and r['term']=='full' and r['mode']!='identity'];model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 buckets=defaultdict(list)
 for r in rows:buckets[r['layer'],len(examples[r['id'],r['receiver']]['ids'])].append(r)
 errs=[];outrel=[]
 for (l,length),rr in buckets.items():
  for lo in range(0,len(rr),12):
   bb=rr[lo:lo+12];vs=[]
   for r in bb:
    u=c['inputs'][r['id'],r['receiver']][l].double();d=c['inputs'][r['id'],r['donor']][l].double();t=d-u;t-=u*(t@u)/u.square().sum();theta=2*math.asin(.3/2);vs.append(math.cos(theta)*u+math.sin(theta)*u.norm()*t/t.norm())
   def pre(m,args):x=args[0].clone();x[:,-1]=torch.stack(vs).to(x);return(x,)
   def post(m,args,out):
    for i,r in enumerate(bb):
     k=(r['id'],r['receiver'],r['donor'],l,.3,'full');pred=c['outputs'][k[:2]][l]+c['patches'][k];outrel.append(float((out[i,-1].cpu()-pred).norm()/pred.norm()))
   hs=[model.blocks[l].mlp.register_forward_pre_hook(pre),model.blocks[l].mlp.register_forward_hook(post)]
   try:out,_=run(model,torch.tensor([examples[r['id'],r['receiver']]['ids'] for r in bb],device='cuda'))
   finally:
    for h in hs:h.remove()
   for i,r in enumerate(bb):
    k=r['id'],r['receiver'];change=measure(out[i].cpu(),examples[k]['groups'])['margin']-measure(c['native'][k],examples[k]['groups'])['margin'];errs.append(abs(change-r['margin_change']))
 result={'direct_input_vs_analytic_output_swap_max_margin_error':max(errs),'direct_input_vs_analytic_output_max_relative_L2':max(outrel),'whole_model_forwards':len(rows),'all_families':True,'radius':.3};assert max(errs)<1e-4 and max(outrel)<1e-5;(ART/f'{s}_direct_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()

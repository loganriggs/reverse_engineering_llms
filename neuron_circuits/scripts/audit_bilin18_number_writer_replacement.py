"""H586 independent overwrite versus additive full-swap audit."""
import json,torch
from collections import defaultdict
from run_bilin18_diverse_causal_tasks import ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
@torch.inference_mode()
def main():
 torch.set_num_threads(6);stem='bilin18_number_subject_attractor';j=json.loads((ART/f'{stem}.json').read_text());data=torch.load(ART/f'{stem}_cache.pt',weights_only=False);ex=data['examples'];w=data['writes'];idx={(e['kind'],e['item'],e['frame'],e['subject_number'],e['attractor_number']):k for k,e in ex.items()};prior={(c['id'],c['swap']):c for c in j['interventions'] if c['mode']=='full'};model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):m.float().eval()
 jobs=defaultdict(list)
 for k,e in ex.items():
  for sw in ('subject','attractor'):
   sn=e['subject_number'];an=e['attractor_number'];dk=idx[e['kind'],e['item'],e['frame'],1-sn if sw=='subject' else sn,1-an if sw=='attractor' else an];jobs[len(e['ids'])].append((k,sw,dk))
 errors=[]
 for T,jj in jobs.items():
  for st in range(0,len(jj),24):
   bb=jj[st:st+24];don=torch.stack([w[dk] for k,sw,dk in bb]).cuda()
   def h(m,args,out):y,bus=out;y=y.clone();y[:,-1]=don.to(y);return y,bus
   handle=model.blocks[5].attn.register_forward_hook(h)
   try:z,_=run(model,torch.tensor([ex[k]['ids'] for k,sw,dk in bb],device='cuda'))
   finally:handle.remove()
   for ix,(k,sw,dk) in enumerate(bb):dm=measure(z[ix],ex[k]['groups'])['margin']-measure(data['native'][k],ex[k]['groups'])['margin'];errors.append(abs(dm-prior[k,sw]['margin_delta']))
 audit=dict(cases=len(errors),overwrite_replay_max_margin_error=max(errors));assert len(errors)==256 and max(errors)<3e-5;(ART/'bilin18_number_writer_replacement_audit.json').write_text(json.dumps(audit,indent=2));print(audit)
if __name__=='__main__':main()

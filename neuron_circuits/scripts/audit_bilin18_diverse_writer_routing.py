"""Independent H581 value-path reconstruction using live c_v and c_proj hooks."""
import json,torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ART
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
@torch.inference_mode()
def main():
 torch.set_num_threads(6);s='bilin18_diverse_writer_transfer';saved=torch.load(ART/f'{s}_cache.pt',weights_only=False);model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):m.float().eval()
 H=model.config.n_head;D=model.config.n_embd//H;errors=[];rel=[]
 for k,e in saved['examples'].items():
  if e['item']!=0 or e['variant']!=0:continue
  ids=torch.tensor([e['ids']],device='cuda');T=ids.shape[1]
  for l,h in ((4,0),(7,8)):
   for site in (-1,-2):
    cap={};att=model.blocks[l].attn
    def ph(m,args):cap['pre']=args[0][0,-1,h*D:(h+1)*D].clone()
    hook=att.c_proj.register_forward_pre_hook(ph)
    run(model,ids);before=cap['pre'].clone()
    donor=saved['captures'][k[0],1][l,'c_v'][site,h].cuda();native=saved['captures'][k][l,'c_v'][site,h].cuda()
    def vh(m,args,out):out=out.clone();out[0,site,h*D:(h+1)*D]=donor;return out
    hh=att.c_v.register_forward_hook(vh)
    try:run(model,ids);after=cap['pre'].clone()
    finally:hh.remove();hook.remove()
    cos,sin=att.rotary(torch.zeros(1,T,H,D,device='cuda'));vv={}
    for n in ('c_q','c_k','c_q2','c_k2'):
     # Full-length replay, contrasting H581's two-position rotary slices.
     x=torch.zeros(1,T,H,D,device='cuda');x[0,-2:]=saved['captures'][k][l,n].cuda();vv[n]=model.tt.apply_rotary_emb(F.rms_norm(x,(D,)),cos,sin)[0,:,h]
    p=(vv['c_q'][-1]@vv['c_k'][site]/D)*(vv['c_q2'][-1]@vv['c_k2'][site]/D)
    predicted=(donor-native)*(1-att.lamb)*p;actual=after-before;errors.append(float((actual-predicted).abs().max()));rel.append(float((actual-predicted).norm()/predicted.norm().clamp_min(1e-8)))
 audit=dict(cases=len(errors),preprojection_max_abs_error=max(errors),preprojection_max_relative_error=max(rel));assert audit['cases']==40 and max(errors)<.001
 (ART/'bilin18_diverse_writer_routing_audit.json').write_text(json.dumps(audit,indent=2));print(audit)
if __name__=='__main__':main()

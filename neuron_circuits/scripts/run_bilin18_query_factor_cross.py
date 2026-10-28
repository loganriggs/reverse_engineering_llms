"""H567: base/edited query factors crossed at edited position with recipient key fixed."""
import json,hashlib,time
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_query_factor_cross'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_equal_length_gap_content.json';old=json.loads(src.read_text());saved=torch.load(ART/'bilin18_equal_length_gap_content_native_cache.pt',weights_only=False);allrows=saved['rows'];rows=[r for r in allrows if r['gap']!='base'];by={r['id']:r for r in allrows};cache=saved['cache'];native=saved['native'];cfg=dict(experiment='H567',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),heads=[[8,3],[14,4]],modes=['base_base','edited_base','base_edited','edited_edited'],design='All 192 edited prompts from H566. Cross the two final-query factors independently between base and edited context, always at edited destination position and recipient source key. Alter only selected source persistent-bus edge; all other computation live.',hypotheses=['One query factor dominates contextual gate correction across tasks/phrases.','Factor roles depend on head/task, or both contribute with cancellation.'],scope='Diagnostic reuse of prior panel. No semantic query labels inferred; local edge hybrids are not full query swaps.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for param in mod.parameters():param.requires_grad_(False)
 H=model.config.n_head;D=model.config.n_embd//H;result=dict(config=cfg,geometry=[],interventions=[],status='running');jobs=defaultdict(list);oldg={(g['id'],g['layer']):g for g in old['geometry']};errors=[]
 for r in rows:
  base=by[r['baseid']+'-base'];c=cache[r['id']];bc=cache[base['id']];T=len(r['ids']);p=r['source_position']
  for l,h in cfg['heads']:
   attn=model.blocks[l].attn;query={};rawkey={n:c[(l,n)][p].cuda().view(1,1,H,D).expand(1,T,H,D) for n in ('c_k','c_k2')};cos,sin=attn.rotary(rawkey['c_k']);keys={n:model.tt.apply_rotary_emb(F.rms_norm(v,(D,)),cos,sin)[0,p,h] for n,v in rawkey.items()}
   for origin,ca in [('base',bc),('edited',c)]:
    query[origin]=[]
    for qn,kn in [('c_q','c_k'),('c_q2','c_k2')]:
     v=ca[(l,qn)][-1].cuda().view(1,1,H,D).expand(1,T,H,D);q=model.tt.apply_rotary_emb(F.rms_norm(v,(D,)),cos,sin)[0,-1,h];query[origin].append(float(q@keys[kn]/D))
   a,b=query['base'];u,v=query['edited'];lam=float(attn.lamb);coefs=dict(base_base=lam*a*b,edited_base=lam*u*b,base_edited=lam*a*v,edited_edited=lam*u*v);actual=coefs['edited_edited'];errors.append(abs(actual-oldg[r['id'],l]['actual_coefficient']));result['geometry'].append(dict(id=r['id'],family=r['family'],gap=r['gap'],layer=l,head=h,base_factors=[a,b],edited_factors=[u,v],coefficients=coefs,first_term=lam*(u-a)*b,second_term=lam*a*(v-b),interaction_term=lam*(u-a)*(v-b),total_change=actual-coefs['base_base']))
   core=F.linear(c['bus'][p,h].cuda(),attn.c_proj.weight[:,h*D:(h+1)*D])
   for mode,coef in coefs.items():jobs[T,l].append((r,h,mode,((coef-actual)*core).cpu()))
 print('LOCKED JOBS',sum(map(len,jobs.values())),flush=True)
 for (T,l),jj in jobs.items():
  for begin in range(0,len(jj),24):
   bb=jj[begin:begin+24];zz,_=run(model,torch.tensor([r['ids'] for r,h,m,d in bb],device='cuda'),patch=(l,torch.stack([d for r,h,m,d in bb])))
   for ix,(r,h,mode,d) in enumerate(bb):result['interventions'].append(dict(id=r['id'],family=r['family'],gap=r['gap'],layer=l,head=h,mode=mode,margin_delta=measure(zz[ix],r['groups'])['margin']-native[r['id']]['margin']))
 result['summary']={}
 for fam in ('binding','copying'):
  result['summary'][fam]={}
  for l,h in cfg['heads']:
   gg=[g for g in result['geometry'] if g['family']==fam and g['layer']==l];entry={k:float(np.mean([abs(g[k]) for g in gg])) for k in ('first_term','second_term','interaction_term','total_change')};entry['modes']={}
   for mode in cfg['modes']:
    cc=[c for c in result['interventions'] if c['family']==fam and c['layer']==l and c['mode']==mode];entry['modes'][mode]=dict(coefficient_mae=float(np.mean([abs(g['coefficients'][mode]-g['coefficients']['edited_edited']) for g in gg])),margin_mae=float(np.mean([abs(c['margin_delta']) for c in cc])))
   result['summary'][fam][f'L{l}H{h}']=entry
 result['audits']=dict(native_coefficient_max_error=max(errors),self_margin_max_error=max(abs(c['margin_delta']) for c in result['interventions'] if c['mode']=='edited_edited'),decomposition_max_error=max(abs(g['first_term']+g['second_term']+g['interaction_term']-g['total_change']) for g in result['geometry']),finite=all(np.isfinite(c['margin_delta']) for c in result['interventions']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(summary=result['summary'],audits=result['audits']),indent=2),flush=True)
if __name__=='__main__':main()

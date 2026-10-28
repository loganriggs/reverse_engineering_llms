"""H569: pooled, mismatched and cross-task controls for learned query edits."""
import json,hashlib,time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_query_edit_controls'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_shared_query_edit_transfer.json';saved=torch.load(ART/'bilin18_shared_query_edit_transfer_native_cache.pt',weights_only=False);cache=saved['cache'];native=saved['native'];rows=[r for r in saved['rows'] if r['gap']!='base'];dp=ART/'bilin18_shared_query_edit_transfer_locked_deltas.pt';deltas=torch.load(dp,weights_only=False);dh=hashlib.sha256(dp.read_bytes()).hexdigest();groups=[['usual','fact','once'],['expected','said','others']];cfg=dict(experiment='H569',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),donor_sha256=dh,heads=[[8,3],[14,4]],modes=['self','phase','matched','pool_all','pool_other_same_length','wrong_a','wrong_b','other_task'],design='Reuse H568 edited test panel and locked discovery means. Change both query projections. Same-length pool excludes target phrase; wrong controls use both other phrases in fixed group order. No new fitting or outcome selection.',scope='Diagnostic controls on previously evaluated prompts. Source-edge substitutions, not complete query replacement. Task labels confounded with template/source role.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 H=model.config.n_head;D=model.config.n_embd//H;result=dict(config=cfg,geometry=[],interventions=[],status='running');jobs=defaultdict(list)
 for r in rows:
  c=cache[r['id']];bc=cache[r['baseid']+'-base'];T=len(r['ids']);p=r['source_position'];group=next(g for g in groups if r['gap'] in g);others=[g for g in group if g!=r['gap']]
  for l,h in cfg['heads']:
   attn=model.blocks[l].attn;dummy=torch.zeros(1,T,H,D,device='cuda');cos,sin=attn.rotary(dummy);keys={}
   for n in ('c_k','c_k2'):
    v=c[l,n][p].cuda().view(1,1,H,D).expand(1,T,H,D);keys[n]=model.tt.apply_rotary_emb(F.rms_norm(v,(D,)),cos,sin)[0,p,h]
   def coef(qs):
    fs=[]
    for qn,kn in [('c_q','c_k'),('c_q2','c_k2')]:
     v=qs[qn].cuda().view(1,1,1,D).expand(1,T,1,D);q=model.tt.apply_rotary_emb(F.rms_norm(v,(D,)),cos,sin)[0,-1,0];fs.append(float(q@keys[kn]/D))
    return fs[0]*fs[1]*float(attn.lamb)
   baseq={n:bc[l,n][-1,h*D:(h+1)*D] for n in ('c_q','c_q2')};actual=coef({n:c[l,n][-1,h*D:(h+1)*D] for n in baseq});core=F.linear(c['bus'][p,h].cuda(),attn.c_proj.weight[:,h*D:(h+1)*D]);donors=dict(matched=deltas[r['family'],r['gap'],l],pool_all={n:torch.stack([deltas[r['family'],g,l][n] for group0 in groups for g in group0]).mean(0) for n in baseq},pool_other_same_length={n:torch.stack([deltas[r['family'],g,l][n] for g in others]).mean(0) for n in baseq},wrong_a=deltas[r['family'],others[0],l],wrong_b=deltas[r['family'],others[1],l],other_task=deltas['copying' if r['family']=='binding' else 'binding',r['gap'],l]);coefs=dict(self=actual,phase=coef(baseq))
   for mode,don in donors.items():coefs[mode]=coef({n:baseq[n]+don[n] for n in baseq})
   for mode,co in coefs.items():result['geometry'].append(dict(id=r['id'],family=r['family'],gap=r['gap'],layer=l,head=h,mode=mode,coefficient=co,actual=actual));jobs[T,l].append((r,h,mode,((co-actual)*core).cpu()))
 print('LOCKED',sum(map(len,jobs.values())),flush=True)
 for (T,l),jj in jobs.items():
  for begin in range(0,len(jj),24):
   bb=jj[begin:begin+24];zz,_=run(model,torch.tensor([r['ids'] for r,h,m,d in bb],device='cuda'),patch=(l,torch.stack([d for r,h,m,d in bb])))
   for ix,(r,h,mode,d) in enumerate(bb):result['interventions'].append(dict(id=r['id'],family=r['family'],gap=r['gap'],layer=l,head=h,mode=mode,margin_delta=measure(zz[ix],r['groups'])['margin']-native[r['id']]['margin']))
 result['summary']={}
 for fam in ('binding','copying'):
  result['summary'][fam]={}
  for l,h in cfg['heads']:
   result['summary'][fam][f'L{l}H{h}']={}
   for mode in cfg['modes']:
    gg=[g for g in result['geometry'] if g['family']==fam and g['layer']==l and g['mode']==mode];cc=[c for c in result['interventions'] if c['family']==fam and c['layer']==l and c['mode']==mode];result['summary'][fam][f'L{l}H{h}'][mode]=dict(coefficient_mae=float(np.mean([abs(g['coefficient']-g['actual']) for g in gg])),margin_mae=float(np.mean([abs(c['margin_delta']) for c in cc])))
 old=json.loads(src.read_text());oldmatched={(c['id'],c['layer']):c['margin_delta'] for c in old['interventions'] if c['mode']=='both_delta'};result['audits']=dict(self_margin_max_error=max(abs(c['margin_delta']) for c in result['interventions'] if c['mode']=='self'),matched_replay_max_error=max(abs(c['margin_delta']-oldmatched[c['id'],c['layer']]) for c in result['interventions'] if c['mode']=='matched'),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),donors_unchanged=dh==hashlib.sha256(dp.read_bytes()).hexdigest(),finite=all(np.isfinite(c['margin_delta']) for c in result['interventions']));result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(summary=result['summary'],audits=result['audits']),indent=2),flush=True)
if __name__=='__main__':main()

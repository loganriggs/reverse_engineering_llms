"""H970: exact attention5 source/head and routing-value factorial accounting."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward
from audit_bilin18_mlp_discriminators import TOKENIZER
from transformers import AutoTokenizer
from pathlib import Path
import torch,json,numpy as np,time,re
S='bilin18_attention5_sources'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'panel':ART/'bilin18_factorial_valence_negation.json','data':ART/'bilin18_factorial_valence_negation_captures.pt','trace':ART/'bilin18_valence_interaction_trace_captures.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);trace=torch.load(paths['trace'],weights_only=False);panel=json.loads(paths['panel'].read_text());ex=data['examples'];qs=data['quartets'];native=data['native'];tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 cfg={'experiment':'H970','hashes':hashes,'design':'Reconstruct attention5 last-token patterns and mixed values including native layer0 bus, QK RMS norm, rotary embeddings, checkpoint-native product of two QK score matrices (signed, unnormalized routing weights) and output projection. Decompose by head and source token. Align tokens by prefix/negation/adjective/suffix roles and within-role index.','factorization':'At each aligned slot/head, attention output is scalar P times projected value V. Factorial interaction = P0*Vi (value interaction) + Pi*V0 (routing interaction) + Pa*Vn+Pn*Va (cross terms). Summed over sources/heads.','alignment':'Role assignment uses tokenizer character offsets. Slots missing in some quartet states use zero P and zero V; factor split on those slots depends on this padding convention. Report common versus noncommon slots separately. Source contributions P*V always sum exactly; padding is not a causal hypothesis.','limits':'Native trajectory factorization, not mediation or a counterfactual routing experiment. Native-gate failures remain diagnostics; rank only competent familiar A. No new training or reserved-test use.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');slots={}
 for i,e in enumerate(ex):
  m=re.search(r'(?:was|as) (not )?([^.]*)\.',e['text']);assert m
  ast,aend=m.span(2);nst=m.start(1) if m.group(1) else ast;enc=tok(e['text'],add_special_tokens=False,return_offsets_mapping=True);assert enc['input_ids']==e['ids'];count={};ss=[]
  for a,b in enc['offset_mapping']:
   role='prefix' if b<=nst else ('negation' if b<=ast else ('adjective' if a<aend and b>ast else 'suffix'))
   ix=count.get(role,0);count[role]=ix+1;ss.append(f'{role}:{ix}')
  assert len(set(ss))==len(ss);slots[i]=ss
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 att=model.blocks[5].attn;assert att.squared_attn and hasattr(att,'c_q2');from jacclust.tt_model import apply_rotary_emb
 buf={}
 def pre(module,args):
  x,bus=args;batch,T,C=x.shape;hh=module.n_head;dd=module.head_dim;q=module.c_q(x).view(batch,T,hh,dd);k=module.c_k(x).view_as(q);v=(1-module.lamb)*module.c_v(x).view_as(q)+module.lamb*bus.view_as(q);cos,sin=module.rotary(q);q=apply_rotary_emb(torch.nn.functional.rms_norm(q,(dd,)),cos,sin);k=apply_rotary_emb(torch.nn.functional.rms_norm(k,(dd,)),cos,sin);score=torch.einsum('bhd,bthd->bht',q[:,-1],k);q2=module.c_q2(x).view_as(q);k2=module.c_k2(x).view_as(k);q2=apply_rotary_emb(torch.nn.functional.rms_norm(q2,(dd,)),cos,sin);k2=apply_rotary_emb(torch.nn.functional.rms_norm(k2,(dd,)),cos,sin);score2=torch.einsum('bhd,bthd->bht',q2[:,-1],k2);P=(score/dd)*(score2/dd);z=torch.einsum('bht,bthd->bhd',P,v);out=module.c_proj(z.reshape(batch,C));buf.update(P=P.detach().cpu(),V=v.detach().cpu(),out=out.detach().cpu())
 h=att.register_forward_pre_hook(pre);captures={};replay=0.;outreplay=0.;actual=0
 try:
  for length in sorted({len(e['ids']) for e in ex}):
   inds=[i for i,e in enumerate(ex) if len(e['ids'])==length]
   for st in range(0,len(inds),24):
    ii=inds[st:st+24];y,_=forward(model,torch.tensor([ex[i]['ids'] for i in ii],device='cuda'));actual+=len(ii)
    for k,i in enumerate(ii):captures[i]={key:buf[key][k].clone() for key in ['P','V']};replay=max(replay,float((y[k].double().cpu()-native[i]).abs().max()));old=trace['captures'][i][5,'attention'];outreplay=max(outreplay,float((buf['out'][k]-old).norm()/old.norm()))
 finally:h.remove()
 W=att.c_proj.weight.detach().double().cpu().reshape(1152,att.n_head,att.head_dim);H=torch.tensor([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],dtype=torch.float64);records=[];factorrecords=[];parts={};factorerr=0.;nativeerr=0.
 for q in qs:
  union=sorted({s for i in q['indices'] for s in slots[i]});common=set.intersection(*[set(slots[i]) for i in q['indices']]);P=torch.zeros(4,len(union),att.n_head,dtype=torch.float64);V=torch.zeros(4,len(union),att.n_head,1152,dtype=torch.float64)
  for state,i in enumerate(q['indices']):
   vproj=torch.einsum('thd,ohd->tho',captures[i]['V'].double(),W)
   for t,slot in enumerate(slots[i]):ix=union.index(slot);P[state,ix]=captures[i]['P'][:,t].double();V[state,ix]=vproj[t]
  Y=P[:,:,:,None]*V;pc=torch.einsum('sc,sth->cth',H,P)/4;vc=torch.einsum('sc,stho->ctho',H,V)/4;yc=torch.einsum('sc,stho->ctho',H,Y)/4;value=pc[0,:,:,None]*vc[3];routing=pc[3,:,:,None]*vc[0];cross=pc[1,:,:,None]*vc[2]+pc[2,:,:,None]*vc[1];total=yc[3].sum((0,1));factorerr=max(factorerr,float((value+routing+cross-yc[3]).norm()/yc[3].norm()));ref=trace['coefficients'][q['id']][5,'attention'][3];nativeerr=max(nativeerr,float((total-ref).norm()/ref.norm()));parts[q['id']]=dict(value=value.sum((0,1)),routing=routing.sum((0,1)),cross=cross.sum((0,1)),total=total)
  den=float(total.square().sum())
  for t,slot in enumerate(union):
   for head in range(att.n_head):
    vec=yc[3,t,head];records.append({k:q[k] for k in ['id','format','cohort','source']}|dict(slot=slot,role=slot.split(':')[0],head=head,common=slot in common,dot_total=float(vec@total),total_squared_norm=den,norm=float(vec.norm())))
  for name,v in [('value',value),('routing',routing),('cross',cross)]:
   for shared in [True,False]:
    tt=[t for t,s in enumerate(union) if (s in common)==shared];vec=v[tt].sum((0,1)) if tt else total*0;factorrecords.append({k:q[k] for k in ['id','format','cohort','source']}|dict(component=name,common=shared,dot_total=float(vec@total),total_squared_norm=den,norm=float(vec.norm())))
 summaries={}
 for groupname,key in [('roles','role'),('heads','head')]:
  rr=[r for r in records if r['format']=='A' and r['cohort']=='familiar_adjectives'];den=sum(parts[q['id']]['total'].square().sum().item() for q in qs if q['format']=='A' and q['cohort']=='familiar_adjectives');summaries[groupname]=[{key:k,'signed_projection_share':sum(r['dot_total'] for r in rr if r[key]==k)/den} for k in sorted({r[key] for r in rr})]
 rr=[r for r in factorrecords if r['format']=='A' and r['cohort']=='familiar_adjectives'];summaries['factors']=[dict(component=name,common=shared,signed_projection_share=sum(r['dot_total'] for r in rr if r['component']==name and r['common']==shared)/den) for name in ['value','routing','cross'] for shared in [True,False]]
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),whole_model_forwards=actual,expected_forwards=len(ex),native_logits_max_abs=replay,attention_output_relative=outreplay,factorial_product_identity_relative=factorerr,total_interaction_native_relative=nativeerr,finite=all(np.isfinite(r['dot_total']) for r in records),no_training=True,test_evaluations=0)
 assert audits['hashes_unchanged'] and actual==len(ex) and replay<1e-4 and outreplay<1e-5 and factorerr<1e-10 and nativeerr<1e-5 and audits['finite']
 torch.save(dict(captures=captures,slots=slots,parts=parts,projection=W),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,competence=panel['competence'],records=records,factor_records=factorrecords,summary=summaries,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(summaries,audits,flush=True)
if __name__=='__main__':main()

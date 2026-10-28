"""H740: L13H0 operation x surface format with coefficient interventions."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch,numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_broad_task_competence import continuation_id
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_static_bus_map import static_bus
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_recall_operation_cross'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'manifest':ART/'bilin18_reader_operation_format_manifest.json','model':SNAPSHOT/'pytorch_model.bin','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py'),'runner':Path(__file__),'static_bus':ROOT/'scripts/run_bilin18_static_bus_map.py','forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={};excluded=[]
 for row in rows:
  ee=[]
  for v,text in enumerate(row['prompts']):
   gs=[[continuation_id(tok,text,' '+a)] for a in row['answers']];ids=tok.encode(text,add_special_tokens=False);ee.append(dict(ids=ids,groups=gs,family=row['family'],format=row['format']))
  if any(g[0] is None for e in ee for g in e['groups']) or len(ee[0]['ids'])!=len(ee[1]['ids']):excluded.append(row['id']);continue
  pos=[i for i,(a,b) in enumerate(zip(ee[0]['ids'],ee[1]['ids'])) if a!=b];assert pos
  for v,e in enumerate(ee):e['source_position']=pos[-1];examples[row['id'],v]=e
 matching=[]
 for fmt in ['narrative','list']:
  for i in range(8):
   for v in [0,1]:
    ak=(f'cross-entity_binding-{fmt}-{i}',v);bk=(f'cross-sequence_copy-{fmt}-{i}',v)
    if ak not in examples or bk not in examples:continue
    a,b=examples[ak],examples[bk];audit=dict(format=fmt,item=i,variant=v,final_position_match=len(a['ids'])==len(b['ids']),source_position_match=a['source_position']==b['source_position'],source_token_match=a['ids'][a['source_position']]==b['ids'][b['source_position']],final_token_match=a['ids'][-1]==b['ids'][-1]);matching.append(audit);assert all(audit[k] for k in ['final_position_match','source_position_match','source_token_match','final_token_match'])
 cfg=dict(experiment='H740',hashes=hashes,design='Reuse48pairH543operation-formatmanifest, evaluatefixedL13H0 newly. Fourprimarycells narrative/list x binding/copying share names, finaltoken,source/finalpositions; verifyexactmatching. Additionalreversedbindinglist andtemporalcopyingnarrative retained. Capture nativecompetence andlastchangedsource coefficientalpha=lambda13*A(source). Interveneonlythis source buscontribution, notallchangedcopypositions.',interventions='Originaldonor source andidentity controls. AbsentpairsKevin/Brian,stone/glass,Canada/Australia each atgate+1,0,-1. Replacementdelta=alpha*(gate*B(absent)-B(native)). Thuszero removesnative sourcecontribution andmustmakebothabsentconditionsidentical; reversalreversesreplacement contribution aroundsamezero-sourcebackground. Downstreamlive; fixedrawmap predicts identitycontrast signs withoutfitting.',limitations='Previouslyused prompts forotherheads; notgloballyfresh. Operation andformat crossed but lexical/circuitstate maystillconfound abstraction. Nativecompetence gates >=4pairs,.75bothendpointchoice,.5fullvocabtop1; retainallcells. Gate reversal isunnatural counterfactual anddoesnotprove native necessity. No reservednaturaltest.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda');assert model.config.squared_attn
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 attn=model.blocks[13].attn;lam=float(attn.lamb);W=attn.c_proj.weight[:,:128];U=model.lm_head.weight;native={};cache={};head_errors=[];buckets=defaultdict(list);forwards=0
 for k,e in examples.items():buckets[len(e['ids'])].append(k)
 for length,keys in buckets.items():
  for lo in range(0,len(keys),12):
   bb=keys[lo:lo+12];cap={};hooks=[]
   def pre(m,args):cap['bus']=args[1].view(len(bb),length,9,128)[:,:,0].cpu()
   hooks.append(attn.register_forward_pre_hook(pre))
   for name in ['c_q','c_k','c_q2','c_k2','c_v']:
    def h(m,args,out,name=name):cap[name]=out[:,:,:128].cpu()
    hooks.append(getattr(attn,name).register_forward_hook(h))
   def ph(m,args):cap['head']=args[0][:,-1,:128].cpu()
   hooks.append(attn.c_proj.register_forward_pre_hook(ph))
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:
    for h in hooks:h.remove()
   forwards+=len(bb);q,k,q2,k2=[cap[n].cuda()[:,:,None,:] for n in ['c_q','c_k','c_q2','c_k2']];cos,sin=attn.rotary(q);q,k,q2,k2=[model.tt.apply_rotary_emb(F.rms_norm(t,(128,)),cos,sin) for t in [q,k,q2,k2]];pattern=((q[:,-1]*k[:,:,0]).sum(-1)/128)*((q2[:,-1]*k2[:,:,0]).sum(-1)/128);V=(1-lam)*cap['c_v'].cuda()+lam*cap['bus'].cuda();headrec=torch.einsum('bt,btd->bd',pattern,V);true=cap['head'].cuda();head_errors.extend(((headrec-true).norm(dim=1)/true.norm(dim=1).clamp_min(1e-12)).tolist())
   for n,key in enumerate(bb):native[key]=logits[n].cpu();cache[key]=dict(pattern=pattern[n].cpu(),bus=cap['bus'][n])
 competence={}
 for family in ['entity_binding','sequence_copy']:
  competence[family]={}
  for fmt in sorted({r['format'] for r in rows if r['family']==family}):
   rr=[r for r in rows if r['family']==family and r['format']==fmt and (r['id'],0) in examples];both=[];top=[]
   for r in rr:
    mm=[measure(native[r['id'],v],examples[r['id'],v]['groups']) for v in [0,1]];both.append(mm[0]['margin']>0 and mm[1]['margin']<0);top.extend(mm[v]['top1'][v] for v in [0,1])
   s=dict(pairs=len(rr),both_endpoint_choice=float(np.mean(both)),full_vocab_top1=float(np.mean(top)));s['pass']=len(rr)>=4 and s['both_endpoint_choice']>=.75 and s['full_vocab_top1']>=.5;competence[family][fmt]=s
 print('COMPETENCE',competence,flush=True);jobs=[];meta={};geometry=[];staticerrors=[];pairs=[('names',[' Kevin',' Brian']),('objects',[' stone',' glass']),('countries',[' Canada',' Australia'])]
 for row in rows:
  for rv in [0,1]:
   key=(row['id'],rv)
   if key not in examples:continue
   e=examples[key];pos=e['source_position'];before=e['ids'][pos];after=examples[row['id'],1-rv]['ids'][pos];vals=static_bus(model,[before,after]);alpha=lam*float(cache[key]['pattern'][pos]);actual=cache[key]['bus'][pos];staticerrors.append(float((vals[0].cpu()-actual).norm()/actual.norm()));geometry.append(dict(id=row['id'],family=row['family'],format=row['format'],receiver=rv,source_position=pos,final_position=len(e['ids'])-1,alpha=alpha))
   for mode,delta in [('task',alpha*(vals[1]-vals[0])),('identity',torch.zeros_like(vals[0]))]:jobs.append(dict(key=key,pair=None,gate=None,variant=None,mode=mode,delta=delta.cpu()))
   text=row['prompts'][rv]
   for pair,words in pairs:
    ids=[continuation_id(tok,text,w) for w in words]
    if any(i is None for i in ids) or any(i in e['ids'] or any(i in g for g in e['groups']) for i in ids):raise ValueError('Fixed absent pair failed token/absence check')
    vv=static_bus(model,ids);alignment=float((U[ids[0]]-U[ids[1]])@(W@(vv[0]-vv[1])));meta[key,pair]=dict(id=row['id'],family=row['family'],format=row['format'],receiver=rv,pair=pair,alpha=alpha,static_alignment=alignment,tokens=ids)
    for gate in [1,0,-1]:
     for v in [0,1]:jobs.append(dict(key=key,pair=pair,gate=gate,variant=v,mode='absent',delta=(alpha*(gate*vv[v]-vals[0])).cpu()))
 result={};task=[];identity=[];buckets=defaultdict(list)
 for job in jobs:buckets[len(examples[job['key']]['ids'])].append(job)
 for length,jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];delta=torch.stack([r['delta'] for r in bb]).cuda()
   def edit(m,args):
    y=args[0].clone();y[:,-1,:128]+=delta.to(y);return (y,)
   hook=attn.c_proj.register_forward_pre_hook(edit)
   try:logits,_=run(model,torch.tensor([examples[r['key']]['ids'] for r in bb],device='cuda'))
   finally:hook.remove()
   forwards+=len(bb)
   for n,r in enumerate(bb):
    key=r['key']
    if r['mode']=='absent':
     a,b=meta[key,r['pair']]['tokens'];result[key,r['pair'],r['gate'],r['variant']]=float(logits[n,a]-logits[n,b])
    else:
     gs=examples[key]['groups'];change=measure(logits[n],gs)['margin']-measure(native[key],gs)['margin'];effect=change*(1 if key[1]==1 else -1)
     if r['mode']=='identity':identity.append(abs(effect))
     else:task.append(dict(id=key[0],receiver=key[1],family=examples[key]['family'],format=examples[key]['format'],donor_directed_margin_change=effect))
 absent=[]
 for (key,pair),m in meta.items():
  for gate in [1,0,-1]:
   effect=result[key,pair,gate,0]-result[key,pair,gate,1];pred=m['alpha']*m['static_alignment']*gate;absent.append(dict(**m,gate=gate,observed_identity_effect=effect,predicted_alignment=pred))
 zero=max(abs(r['observed_identity_effect']) for r in absent if r['gate']==0);audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},native_head_factorization_max_relative=max(head_errors),static_bus_replay_max_relative=max(staticerrors),identity_margin_max_abs=max(identity),zero_gate_identity_contrast_max_abs=zero,primary_matched_contexts=len(matching),finite=all(np.isfinite(r['observed_identity_effect']) for r in absent),whole_model_forwards=forwards,test_evaluations=0);print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(head_errors)<1e-4 and max(staticerrors)<1e-5 and max(identity)<1e-4 and zero<1e-4;torch.save(dict(native_logits=native,components=cache,examples=examples),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,excluded_pairs=excluded,matching=matching,competence=competence,geometry=geometry,task=task,absent=absent,audits=audits,elapsed_seconds=time.time()-start),indent=2))
if __name__=='__main__':main()

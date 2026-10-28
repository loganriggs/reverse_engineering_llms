"""H785 predeclared angular, final-position MLP interventions across eight tasks."""
import json,time,math
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,suite,groups,measure
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_size_controlled_bilinear_transfer'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');rows=[r for r in suite() if r['split']=='transfer'];prefix='Complete the following sentence:\n'
 for r in rows:r['prompts']=[prefix+p for p in r['prompts']];r['nuisance']=prefix+r['nuisance']
 manifest=ART/f'{STEM}_manifest.json';manifest.write_text(json.dumps(rows,indent=2)+'\n');cfg={'experiment':'H785','manifest_sha256':sha(manifest),'layers':[1,4],'radii':[.05,.15,.3],'prediction':'At matched angular displacement, L1 has greater first-order output approximation error than L4; evaluate per task, without selecting successful examples. This is a hypothesis, not an algebraic requirement.','design':'Fresh prefixed contexts from all eight existing transfer families. Capture final-position normalized MLP input. Rotate on its constant-norm sphere toward donor tangent direction with chord/input norm rho. Compare full bilinear output change and first-order approximation through live downstream model. Task reciprocal and same-answer nuisance edits; identity zero controls.','limitations':'New prompt contexts but reused base vocabulary/items. Final position differs from H784 value source positions. Off-native intervened states, not a recovered feature or manifold. Competence gates limit semantic interpretation; all families retained.','gate':'At least 4 pairs, >=.75 both-endpoint contrast correctness and >=.5 pooled intended full-vocab top1','fineweb_new_documents':0};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={};excluded=[]
 for r in rows:
  ee=[]
  for v,text in enumerate(r['prompts']+[r['nuisance']]):
   gg=[[continuation_id(tok,text,c) for c in g] for g in groups(r['family'],r['answers'])];gg=[[t for t in g if t is not None] for g in gg]
   if any(not g for g in gg) or set(gg[0])&set(gg[1]):break
   ee.append({'ids':tok.encode(text,add_special_tokens=False),'groups':gg,'family':r['family']})
  if len(ee)!=3:excluded.append(r['id']);continue
  for v,e in enumerate(ee):examples[r['id'],v]=e
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native={};inputs={};outputs={};saved={};handles=[];forwards=0
 for l in [1,4]:
  assert not model.blocks[l].mlp.config.gated
  def pre(m,args,l=l):saved[l,'input']=args[0][:,-1].cpu()
  def post(m,args,out,l=l):saved[l,'output']=out[:,-1].cpu()
  handles.extend([model.blocks[l].mlp.register_forward_pre_hook(pre),model.blocks[l].mlp.register_forward_hook(post)])
 buckets=defaultdict(list)
 for k,e in examples.items():buckets[len(e['ids'])].append(k)
 try:
  for kk in buckets.values():
   for lo in range(0,len(kk),12):
    bb=kk[lo:lo+12];out,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'));forwards+=len(bb)
    for i,k in enumerate(bb):native[k]=out[i].cpu();inputs[k]={l:saved[l,'input'][i] for l in [1,4]};outputs[k]={l:saved[l,'output'][i] for l in [1,4]}
 finally:
  for h in handles:h.remove()
 competence=[]
 for fam in sorted({e['family'] for e in examples.values()}):
  ids=[r['id'] for r in rows if r['family']==fam and (r['id'],0) in examples];both=[];top=[];nuis=[]
  for rid in ids:
   mm=[measure(native[rid,v],examples[rid,v]['groups']) for v in [0,1,2]];both.append(mm[0]['margin']>0 and mm[1]['margin']<0);top.extend([mm[v]['top1'][v] for v in [0,1]]);nuis.append(mm[2]['margin']>0)
  avg=lambda x:sum(x)/len(x) if x else 0.;competence.append({'family':fam,'pairs':len(ids),'both_endpoint_choice':avg(both),'full_vocab_top1':avg(top),'nuisance_choice':avg(nuis),'pass':len(ids)>=4 and avg(both)>=.75 and avg(top)>=.5})
 print('competence',json.dumps(competence),flush=True);patches={};geometry=[];jobs=[];normerr=[];radiuserr=[];replay=[];identity=[];skipped=[]
 for l in [1,4]:
  mlp=model.blocks[l].mlp;WL=mlp.Left.weight.cpu().double();WR=mlp.Right.weight.cpu().double();WD=mlp.Down.weight.cpu().double()
  for rid in sorted({k[0] for k in examples}):
   for rv,dv,mode in [(0,1,'task'),(1,0,'task'),(0,2,'nuisance'),(0,0,'identity')]:
    k=rid,rv;dk=rid,dv;u=inputs[k][l].double();d=inputs[dk][l].double();un=u.norm();t=d-u;t=t-u*(t@u)/(un*un)
    if mode!='identity' and t.norm()<1e-10:skipped.append({'id':rid,'receiver':rv,'donor':dv,'layer':l});continue
    for rho in cfg['radii']:
     theta=2*math.asin(rho/2);v=u if mode=='identity' else math.cos(theta)*u+math.sin(theta)*un*t/t.norm();du=v-u;normerr.append(float(abs(v.norm()/un-1)));radiuserr.append(float(abs(du.norm()/un-(0 if mode=='identity' else rho))));a=WL@u;b=WR@u;da=WL@du;db=WR@du;linear=WD@(da*b+a*db);interaction=WD@(da*db);full=linear+interaction
     analytic=(WD@((WL@v)*(WR@v)))+mlp.Down_bias.cpu().double();replay.append(float((outputs[k][l].double()+full-analytic).norm()/analytic.norm().clamp_min(1e-12)))
     geometry.append({'id':rid,'family':examples[k]['family'],'receiver':rv,'donor':dv,'mode':mode,'layer':l,'rho':rho,'full_squared_norm':float(full.square().sum()),'interaction_squared_norm':float(interaction.square().sum()),'linear_squared_norm':float(linear.square().sum()),'displacement_norm':float(du.norm())})
     for term,patch in [('linear',linear),('full',full)]:
      key=(rid,rv,dv,l,rho,term);patches[key]=patch.float();jobs.append({'key':key,'mode':mode,'family':examples[k]['family']})
 print('geometry complete',len(jobs),flush=True);buckets=defaultdict(list)
 for j in jobs:buckets[j['key'][3],len(examples[j['key'][:2]]['ids'])].append(j)
 records=[]
 for (l,length),jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12]
   def patch(m,args,out):
    y=out.clone()
    for i,j in enumerate(bb):y[i,-1]+=patches[j['key']].to(y)
    return y
   h=model.blocks[l].mlp.register_forward_hook(patch)
   try:out,_=run(model,torch.tensor([examples[j['key'][:2]]['ids'] for j in bb],device='cuda'))
   finally:h.remove()
   forwards+=len(bb)
   for i,j in enumerate(bb):
    rid,rv,dv,l,rho,term=j['key'];k=rid,rv;p=out[i].cpu();records.append({'id':rid,'receiver':rv,'donor':dv,'layer':l,'rho':rho,'term':term,'mode':j['mode'],'family':j['family'],'margin_change':measure(p,examples[k]['groups'])['margin']-measure(native[k],examples[k]['groups'])['margin']})
    if j['mode']=='identity':identity.append(float((p-native[k]).abs().max()))
 audits={'hashes_unchanged':sha(manifest)==cfg['manifest_sha256'],'constant_norm_max_relative_error':max(normerr),'chord_radius_max_abs_error':max(radiuserr),'analytic_output_max_relative_L2':max(replay),'identity_logits_max_abs':max(identity),'finite':all(math.isfinite(r['margin_change']) for r in records),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(normerr+radiuserr)<1e-12 and max(replay)<1e-5 and max(identity)<1e-4
 torch.save({'inputs':inputs,'outputs':outputs,'native':native,'examples':examples,'patches':patches},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'competence':competence,'excluded':excluded,'skipped':skipped,'geometry':geometry,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps(audits),flush=True)
if __name__=='__main__':main()

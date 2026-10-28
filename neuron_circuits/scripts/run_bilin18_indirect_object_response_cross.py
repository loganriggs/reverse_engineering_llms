"""H810 frozen IOI L2H5 perturbation / L6 MLP input-baseline response crossing."""
import json,time,math
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
S='bilin18_indirect_object_response_cross'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'lock':ROOT/'configs/bilin18_indirect_object_mechanism_locked.json','capture':ART/'bilin18_indirect_object_competence_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};lock=json.loads(paths['lock'].read_text());cfg={'experiment':'H810','hashes':hashes,'lock':lock,'design':'Inject delta*projected first-name mixed V at second initial-name L2 attention output. Pair opposite givers; cross L6 raw baseline and finite input response. All four module responses added to own MLP baseline. Fixed-name logprob and logit margin readouts.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');cap=torch.load(paths['capture'],weights_only=False);ex={k:v['example'] for k,v in cap.items()};idx={(e['format'],e['item'],e['order'],e['query']):k for k,e in ex.items()};other={k:idx[e['format'],e['item'],e['order'],1-e['query']] for k,e in ex.items()}
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 W=model.blocks[2].attn.c_proj.weight[:,640:768].cpu().double();directions={k:W@v['value'].double() for k,v in cap.items()};direrr=max(float((directions[k]-directions[other[k]]).abs().max()) for k in cap)
 jobs=[{'id':k,'delta':de} for k in cap for de in lock['amplitudes']];buckets=defaultdict(list)
 for j in jobs:buckets[len(ex[j['id']]['ids'])].append(j)
 saved={};captures={};logits={};source={};normerr=[];nativeerr=[];forwards=0
 def raw2(m,args):saved['raw2']=m.lambdas[0]*args[0]+m.lambdas[1]*args[2]
 def inject(m,args,out):
  y=out[0].clone()
  for i,j in enumerate(bb):
   p=ex[j['id']]['positions'][1];y[i,p]+=j['delta']*directions[j['id']].to(y)
  saved['source']=torch.stack([(saved['raw2']+y)[i,ex[j['id']]['positions'][1]] for i,j in enumerate(bb)]).cpu();return y,out[1]
 def raw6(m,args):saved['raw6']=m.lambdas[0]*args[0]+m.lambdas[1]*args[2]
 def premlp(m,args,out):saved['premlp']=(saved['raw6']+out[0])[:,-1].clone()
 def mlppre(m,args):saved['normalized']=args[0][:,-1].clone()
 def mlpout(m,args,out):saved['mlpout']=out[:,-1].clone()
 for jj in buckets.values():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];hs=[model.blocks[2].register_forward_pre_hook(raw2),model.blocks[2].attn.register_forward_hook(inject),model.blocks[6].register_forward_pre_hook(raw6),model.blocks[6].attn.register_forward_hook(premlp),model.blocks[6].mlp.register_forward_pre_hook(mlppre),model.blocks[6].mlp.register_forward_hook(mlpout)]
   try:out,_=run(model,torch.tensor([ex[j['id']]['ids'] for j in bb],device='cuda'))
   finally:
    for h in hs:h.remove()
   forwards+=len(bb);normerr.append(float((F.rms_norm(saved['premlp'],(1152,))-saved['normalized']).abs().max()))
   for i,j in enumerate(bb):
    k=j['id'],j['delta'];captures[k]={n:saved[n][i].cpu() for n in ['premlp','normalized','mlpout']};logits[k]=out[i].cpu();source[k]=saved['source'][i]
    if j['delta']==0:nativeerr.append(float((out[i].cpu()-cap[j['id']]['logits']).abs().max()))
 sourceerr=max(float((v-source[other[k],de]).abs().max()) for (k,de),v in source.items())
 mlp=model.blocks[6].mlp;assert not mlp.config.gated
 L,R,D=[getattr(mlp,n).weight.double() for n in ['Left','Right','Down']];targets={};expansion=[];reconstruction=[];endpoint_scaled=[];geometry=[]
 for k in ex:
  for amp in [.25,.5]:
   for arm in lock['arms']:
    bk=other[k] if arm in ['baseline_swap','both'] else k;dk=other[k] if arm in ['input_swap','both'] else k;b=captures[bk,0.]['premlp'].cuda();dx=(captures[dk,amp]['premlp']-captures[dk,0.]['premlp']).cuda();u=F.rms_norm(b,(1152,));v=F.rms_norm(b+dx,(1152,));du=v.double()-u.double();ud=u.double();direct=F.linear(F.linear(v.double(),L)*F.linear(v.double(),R),D)-F.linear(F.linear(u.double(),L)*F.linear(u.double(),R),D);lin=F.linear(F.linear(ud,L)*F.linear(du,R)+F.linear(du,L)*F.linear(ud,R),D);quad=F.linear(F.linear(du,L)*F.linear(du,R),D);expansion.append(float((direct.double()-lin-quad).norm()/direct.norm().clamp_min(1e-20)));targets[k,amp,arm]=direct.cpu();geometry.append({'id':k,'amplitude':amp,'arm':arm,'response_norm':float(direct.norm()),'quadratic_norm':float(quad.norm())})
    if arm in ['own','both']:
     expected=captures[bk,amp]['mlpout']-captures[bk,0.]['mlpout'];reconstruction.append(float((direct.cpu()-expected).norm()/expected.norm().clamp_min(1e-20)));endpoint_scaled.append(float((direct.cpu()-expected).norm()/captures[bk,amp]['mlpout'].norm().clamp_min(1e-20)))
 records=[];zeroerr=[];ownerr=[]
 for arm in lock['arms']:
  for jj in buckets.values():
   for lo in range(0,len(jj),12):
    bb=jj[lo:lo+12]
    def patch(m,args,out):
     y=out.clone()
     for i,j in enumerate(bb):
      k=j['id'];de=j['delta'];base=captures[k,0.]['mlpout'];response=targets[k,de,arm] if de else torch.zeros_like(base);y[i,-1]=(base+response).to(y)
     return y
    hs=[model.blocks[2].register_forward_pre_hook(raw2),model.blocks[2].attn.register_forward_hook(inject),model.blocks[6].mlp.register_forward_hook(patch)]
    try:out,_=run(model,torch.tensor([ex[j['id']]['ids'] for j in bb],device='cuda'))
    finally:
     for h in hs:h.remove()
    forwards+=len(bb)
    for i,j in enumerate(bb):
     k=j['id'];de=j['delta'];e=ex[k];z=out[i].cpu().double();base=logits[k,0.].double();p=z.log_softmax(0);p0=base.log_softmax(0);a,b=e['candidate_ids'];t=e['candidate_ids'][e['query_display_index']];records.append({'id':k,'format':e['format'],'item':e['item'],'order':e['order'],'first_name_correct':e['query_display_index']==0,'delta':de,'arm':arm,'first_logprob_response':float(p[a]-p0[a]),'margin_response':float((z[a]-z[b])-(base[a]-base[b])),'correct_logprob_response':float(p[t]-p0[t]),'KL':float((p0.exp()*(p0-p)).sum()),'correct':int(z.argmax())==t})
     if de==0:zeroerr.append(float((z-base).abs().max()))
     if arm=='own':ownerr.append(float((z-logits[k,de].double()).abs().max()))
  print('finished',arm,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'projected_direction_query_max_abs':direrr,'L2_source_state_query_max_abs':sourceerr,'H809_native_full_logits_max_abs':max(nativeerr),'RMS_input_max_abs':max(normerr),'bilinear_expansion_max_relative':max(expansion),'module_reconstruction_max_relative':max(reconstruction),'module_reconstruction_endpoint_scaled':max(endpoint_scaled),'zero_delta_full_logits_max_abs':max(zeroerr),'own_full_logits_max_abs':max(ownerr),'finite':all(math.isfinite(r['first_logprob_response']) and math.isfinite(r['margin_response']) for r in records),'whole_model_forwards':forwards,'test_evaluations':0};print(audits,flush=True);assert audits['hashes_unchanged'] and audits['finite'] and direrr==0 and sourceerr==0 and max(nativeerr+normerr+zeroerr+ownerr)<1e-4 and max(expansion)<1e-8 and max(endpoint_scaled)<1e-5
 torch.save({'captures':captures,'targets':targets,'examples':ex},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'geometry':geometry,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits)
if __name__=='__main__':main()

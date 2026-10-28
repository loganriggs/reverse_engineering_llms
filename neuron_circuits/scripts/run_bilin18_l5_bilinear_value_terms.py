"""H784 exact bilinear factor decomposition and isolated local-value causal readouts."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_bilinear_value_terms'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'native':ART/'bilin18_l5_copy_lexical_transfer_captures.pt','factors':ART/'bilin18_l5_routing_value_decomposition_captures.pt','streams':ART/'bilin18_l5_value_position_streams_captures.pt','roles':ART/'bilin18_l5_value_source_roles.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cap=torch.load(paths['native'],weights_only=False);factors=torch.load(paths['factors'],weights_only=False)['states'];stream=torch.load(paths['streams'],weights_only=False);roles=json.loads(paths['roles'].read_text());positions={(r['id'],r['variant']):r['positions']['value_first']+r['positions']['value_second'] for r in roles['role_manifest']};examples=cap['examples'];native=cap['native_logits'];lam=stream['lambda'];sites=[(l,'mlp') for l in [1,4]];cfg={'experiment':'H784','hashes':hashes,'sites':sites,'design':'Exact factor decomposition at L1/L4, receiver output plus each delta term; first pass replaces donor attention/MLP output at both copied-value positions at one layer0–4site; all other upstream computation native. Capture resulting L5H5 local values. Second fresh receiver pass injects only native-receiver-routing times the weighted local-value change at those two sources into final L5H5 output. Compare to full donor local-value-word target and zero. Task reciprocal,nuisance,identity across lexical cohorts; same alignment exclusions.','limitations':'Sufficiency-style site substitution along isolated value readout, not necessity or unique mediation fraction. First-pass upstream changes can propagate through many paths before L5. Does not preserve upstream bypass effects in second pass. Individual sites can substitute/cancel; do not sum causal fractions.'};cfg['design']='Capture native L1/L4 MLP factors at both value positions. Decompose output delta as Down(deltaLeft*Right + Left*deltaRight + deltaLeft*deltaRight). Add isolated terms, linear sum, full sum, or native output difference to receiver MLP outputs; measure resulting L5H5 local-value readout in a separate pristine receiver run. Audit RMS normalization delta as raw-direction plus scale change.';cfg['limitations']='Receiver-anchored factor decomposition depends on expansion point and Left/Right parameterization. Individual terms are hybrid interventions; their effects need not sum. Reused exploratory copying panel; not a manifold or whole-task necessity test.';(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 saved={};upstream={};batchkeys=[];handles=[];forwards=0;nativeerr=[];localerr=[]
 algebra={}; raw_base={}; norm_errors=[]; norm_geometry=[]; reconstruction=[]; relative_reconstruction=[]; algebra_identity=[]
 for l,kind in sites:
  assert not model.blocks[l].mlp.config.gated
  def blockpre(m,args,l=l):raw_base[l]=m.lambdas[0]*args[0]+m.lambdas[1]*args[2]
  def attncap(m,args,out,l=l):saved[l,'raw']=torch.stack([(raw_base[l]+out[0])[i,positions[k]] for i,k in enumerate(batchkeys)]).cpu()
  def pre(m,args,l=l):saved[l,'normalized']=torch.stack([args[0][i,positions[k]] for i,k in enumerate(batchkeys)]).cpu()
  def capture(m,args,out,l=l,kind=kind):saved[l,kind]=torch.stack([out[i,positions[k]] for i,k in enumerate(batchkeys)]).cpu()
  handles.extend([model.blocks[l].register_forward_pre_hook(blockpre),model.blocks[l].attn.register_forward_hook(attncap),model.blocks[l].mlp.register_forward_pre_hook(pre),model.blocks[l].mlp.register_forward_hook(capture)])
  for factor in ['Left','Right']:
   def factorcap(m,args,out,l=l,factor=factor):saved[l,factor]=torch.stack([out[i,positions[k]] for i,k in enumerate(batchkeys)]).cpu()
   handles.append(getattr(model.blocks[l].mlp,factor).register_forward_hook(factorcap))
 def localcap(m,args,out):saved['local']=torch.stack([out[i,positions[k],640:768] for i,k in enumerate(batchkeys)]).cpu()
 handles.append(model.blocks[5].attn.c_v.register_forward_hook(localcap));buckets=defaultdict(list)
 for k,e in examples.items():buckets[len(e['ids'])].append(k)
 try:
  for length,kk in buckets.items():
   for lo in range(0,len(kk),12):
    batchkeys=kk[lo:lo+12];out,_=run(model,torch.tensor([examples[k]['ids'] for k in batchkeys],device='cuda'));forwards+=len(batchkeys)
    for i,k in enumerate(batchkeys):algebra[k]={l:{name:saved[l,name][i] for name in ['raw','normalized','Left','Right']} for l in [1,4]};upstream[k]={site:saved[site][i] for site in sites};nativeerr.append(float((out[i].cpu()-native[k]).abs().max()));localerr.append(float((saved['local'][i].double()-stream['streams'][k]['local'][positions[k]]).abs().max()))
 finally:
  for h in handles:h.remove()
 assert max(nativeerr+localerr)<1e-4
 torch.save({'upstream':upstream,'positions':positions,'algebra':algebra},ART/f'{STEM}_captures.pt');edits=[];excluded=[];patches={};geometry=[]
 for rid in sorted({k[0] for k in examples}):
  for rv,dv,mode in [(0,1,'task'),(1,0,'task'),(0,2,'nuisance'),(0,0,'identity')]:
   key=rid,rv;donor=rid,dv;e=examples[key];de=examples[donor]
   if len(e['ids'])!=len(de['ids']) or positions[key]!=positions[donor]:excluded.append({'id':rid,'receiver':rv,'donor':dv,'reason':'sequence/value-position alignment'});continue
   q={'key':key,'donor':donor,'mode':mode};edits.append(q);A=factors[key]['A'][-1,positions[key]];target=(1-lam)*A@(stream['streams'][donor]['local'][positions[donor]]-stream['streams'][key]['local'][positions[key]]);patches[key,donor,'target']=target;patches[key,donor,'zero']=torch.zeros_like(target)
 identitylocal=[];replacement_errors=[];terms={};names=['zero','target'];term_geometry=[]
 for q in edits:
  key=q['key'];donor=q['donor']
  for l in [1,4]:
   r=algebra[key][l];d=algebra[donor][l];L=r['Left'].double();V=r['Right'].double();dl=d['Left'].double()-L;dv=d['Right'].double()-V;W=model.blocks[l].mlp.Down.weight.detach().cpu().double()
   tt={'left':(dl*V)@W.T,'right':(L*dv)@W.T,'interaction':(dl*dv)@W.T};tt['linear']=tt['left']+tt['right'];tt['full']=tt['linear']+tt['interaction'];actual=upstream[donor][l,'mlp'].double()-upstream[key][l,'mlp'].double();reconstruction.append(float((tt['full']-actual).abs().max()));relative_reconstruction.append(float((tt['full']-actual).norm()/actual.norm().clamp_min(1e-12)));analytic=(d['Left'].double()*d['Right'].double()-L*V)@W.T;algebra_identity.append(float((tt['full']-analytic).norm()/analytic.norm().clamp_min(1e-12)));tt['native_full']=actual
   xr=r['raw'].double();xd=d['raw'].double();eps=torch.finfo(torch.float32).eps;rr=(xr.square().mean(-1,keepdim=True)+eps).sqrt();rd=(xd.square().mean(-1,keepdim=True)+eps).sqrt();direction=(xd-xr)/rr;scale=xd*(1/rd-1/rr);du=d['normalized'].double()-r['normalized'].double();norm_errors.append(float((direction+scale-du).abs().max()));norm_geometry.append({'id':key[0],'receiver':key[1],'donor':donor[1],'mode':q['mode'],'layer':l,'delta_squared_norm':float(du.square().sum()),'direction_squared_norm':float(direction.square().sum()),'scale_squared_norm':float(scale.square().sum()),'direction_scale_dot':float((direction*scale).sum())})
   for name,t in tt.items():
    terms[key,donor,f'L{l}_{name}']=t;term_geometry.append({'id':key[0],'receiver':key[1],'donor':donor[1],'mode':q['mode'],'layer':l,'term':name,'squared_norm':float(t.square().sum()),'target_squared_norm':float(actual.square().sum())})
 print('decomposition audit',max(reconstruction),max(norm_errors),flush=True);assert max(relative_reconstruction)<1e-5 and max(algebra_identity)<1e-10 and max(norm_errors)<1e-4
 for l in [1,4]:
  for term in ['left','right','interaction','linear','full','native_full']:
   name=f'L{l}_{term}';names.append(name);buckets=defaultdict(list)
   for q in edits:buckets[len(examples[q['key']]['ids'])].append(q)
   for length,jj in buckets.items():
    for lo in range(0,len(jj),12):
     bb=jj[lo:lo+12];perturbed={}
     def swap(m,args,out):
      y=out.clone()
      for i,q in enumerate(bb):
       desired=upstream[q['key']][l,'mlp'].double()+terms[q['key'],q['donor'],name];y[i,positions[q['key']]]=desired.to(y);replacement_errors.append(float((y[i,positions[q['key']]].cpu()-desired.float()).abs().max()))
      return y
     def capturelocal(m,args,out):perturbed['local']=torch.stack([out[i,positions[q['key']],640:768] for i,q in enumerate(bb)]).cpu().double()
     hs=[model.blocks[l].mlp.register_forward_hook(swap),model.blocks[5].attn.c_v.register_forward_hook(capturelocal)]
     try:run(model,torch.tensor([examples[q['key']]['ids'] for q in bb],device='cuda'))
     finally:
      for h in hs:h.remove()
     forwards+=len(bb)
     for i,q in enumerate(bb):
      key=q['key'];donor=q['donor'];delta=perturbed['local'][i]-stream['streams'][key]['local'][positions[key]];A=factors[key]['A'][-1,positions[key]];patch=(1-lam)*A@delta;patches[key,donor,name]=patch;target=patches[key,donor,'target'];geometry.append({'id':key[0],'receiver':key[1],'donor':donor[1],'mode':q['mode'],'cohort':examples[key]['lexical_cohort'],'site':name,'patch_squared_norm':float(patch.square().sum()),'target_squared_norm':float(target.square().sum()),'error_squared_norm':float((patch-target).square().sum())})
      if q['mode']=='identity':identitylocal.append(float(delta.abs().max()))
   print('traced',name,'forwards',forwards,flush=True)
 jobs=[{**q,'site':n} for q in edits for n in names];buckets=defaultdict(list)
 for q in jobs:buckets[len(examples[q['key']]['ids'])].append(q)
 records=[];identity=[]
 for length,jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];patch=torch.stack([patches[q['key'],q['donor'],q['site']] for q in bb]).cuda()
   def readout(m,args):y=args[0].clone();y[:,-1,640:768]+=patch.to(y);return(y,)
   h=model.blocks[5].attn.c_proj.register_forward_pre_hook(readout)
   try:out,_=run(model,torch.tensor([examples[q['key']]['ids'] for q in bb],device='cuda'))
   finally:h.remove()
   forwards+=len(bb)
   for i,q in enumerate(bb):
    key=q['key'];e=examples[key];p=out[i].cpu();records.append({'id':key[0],'receiver':key[1],'donor':q['donor'][1],'mode':q['mode'],'cohort':e['lexical_cohort'],'format':e['format'],'site':q['site'],'margin_change':measure(p,e['groups'])['margin']-measure(native[key],e['groups'])['margin']})
    if q['mode']=='identity' or q['site']=='zero':identity.append(float((p-native[key]).abs().max()))
 refs={(r['id'],r['receiver'],r['donor']):r for r in records if r['site']=='target'};oldidx={(r['id'],r['receiver'],r['donor']):r for r in roles['records'] if r['term']=='local_value_words'};replay=[abs(r['margin_change']-oldidx[k]['margin_change']) for k,r in refs.items()];groups=defaultdict(list)
 for r in records:
  if r['mode']!='identity':groups[r['cohort'],r['mode'],r['site']].append(r)
 summary=[]
 for (cohort,mode,site),rr in groups.items():
  ff=[refs[r['id'],r['receiver'],r['donor']] for r in rr];rms=(sum(r['margin_change']**2 for r in ff)/len(ff))**.5;err=(sum((r['margin_change']-f['margin_change'])**2 for r,f in zip(rr,ff))/len(rr))**.5;summary.append({'cohort':cohort,'mode':mode,'site':site,'target_margin_RMS':rms,'site_margin_RMS':(sum(r['margin_change']**2 for r in rr)/len(rr))**.5,'absolute_RMS_error':err,'relative_RMS_error':err/rms if rms>=.25 else None})
 audits={'bilinear_output_delta_reconstruction_max_abs':max(reconstruction),'bilinear_output_delta_max_relative_L2':max(relative_reconstruction),'double_precision_identity_max_relative_L2':max(algebra_identity),'normalization_delta_reconstruction_max_abs':max(norm_errors),'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_replay_max_abs':max(nativeerr),'local_capture_replay_max_abs':max(localerr),'donor_output_replacement_max_abs':max(replacement_errors),'identity_first_pass_local_max_abs':max(identitylocal),'identity_second_pass_logit_max_abs':max(identity),'H780_target_margin_replay_max_abs':max(replay),'finite':all(torch.isfinite(torch.tensor(r['margin_change'])) for r in records),'aligned_edits':len(edits),'excluded_edits':len(excluded),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(nativeerr+localerr+replacement_errors+identitylocal+identity+replay)<1e-4
 torch.save({'patches':patches,'geometry':geometry},ART/f'{STEM}_paths.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'summary':summary,'geometry':geometry,'term_geometry':term_geometry,'normalization_geometry':norm_geometry,'excluded':excluded,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps(audits,indent=2),flush=True)
if __name__=='__main__':main()

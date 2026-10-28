"""H880 matched-context native MLP Jacobian interchange."""
from run_bilin18_final_scale_clamp import *
S='bilin18_mlp_jacobian_interchange'
MODES=['live','linearized','both_copy','both_capital','swapped'];SCALES=[0.25,1.0]
def normalized_mlp(x,w):
 L,R,D,b=w;r=(x.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();u=x/r
 return (u@L.T*(u@R.T))@D.T+b
def jvp(x,v,w):
 L,R,D,b=w;r=(x.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();u=x/r;du=v/r-x*(x*v).mean(-1,keepdim=True)/r**3
 return ((du@L.T)*(u@R.T)+(u@L.T)*(du@R.T))@D.T
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'panel':ART/'bilin18_reader_recombination_panel_captures.pt','screen':ART/'bilin18_reader_recombination_panel.json','live':ART/'bilin18_reader_gain_confirmation.json','finals':ART/'bilin18_final_scale_clamp_final_states.pt','linearized':ART/'bilin18_downstream_mlp_linearization_final_states.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H880','hashes':hashes,'modes':MODES,'scales':SCALES,'design':'L8H3 writes at0.25 and1 amplitude. Interchange L8–17 native MLP Jacobians including RMS across matched copy/capital contexts; retain recipient native output and subtract recipient native input. Both-copy, both-capital, swapped, self and live conditions; attention live. All192 reused endpoints, no reserved test.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 selfprior=torch.load(paths['linearized'],weights_only=False);data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];prior=torch.load(paths['finals'],weights_only=False);live=json.loads(paths['live'].read_text());previous={(r['id'],r['order'],r['layer'],r['head'],r['mode']):r for r in live['records']};screen=json.loads(paths['screen'].read_text());tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 idsby={i:{typ:[continuation_id(tok,e['text'],' '+v[part]) for v in screen['config']['pairs'][e['pair']]] for typ,part in [('country',0),('capital',1)]} for i,e in enumerate(ex)};index={(e['id'],e['order']):i for i,e in enumerate(ex)};matched={(e['pair'],e['format'],e['order'],e['family']):i for i,e in enumerate(ex)};eps=torch.finfo(torch.float32).eps;radii=torch.stack([(prior[i,8,3,0].square().mean()+eps).sqrt() for i in range(len(ex))]);shared=torch.stack([(radii[i]*radii[matched[e['pair'],e['format'],e['order'],'capital' if e['family']=='copy' else 'copy']]).sqrt() for i,e in enumerate(ex)])
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 U=model.lm_head.weight.double().cpu();bias=None if model.lm_head.bias is None else model.lm_head.bias.double().cpu();skip=float(np.prod([float(b.lambdas[0]) for b in model.blocks[9:]]));cache={};state={'capture':True,'mode':'live'};handles=[];stash={}
 weights={k:tuple(t.detach().double() for t in [b.mlp.Left.weight,b.mlp.Right.weight,b.mlp.Down.weight,b.mlp.Down_bias]) for k,b in enumerate(model.blocks) if k>=8};raw={};jvp_err=0.;fd_err=0.;formula_err=0.;formula_rel=0.
 def prehook(k):
  def hook(mod,args):raw[k]=mod.lambdas[0]*args[0][:,-1]+mod.lambdas[1]*args[2][:,-1]
  return hook
 def atthook(k):
  def hook(mod,args,out):raw[k]=raw[k]+out[0][:,-1]
  return hook
 def mlphook(k):
  def hook(mod,args,out):
   nonlocal jvp_err,fd_err,formula_err,formula_rel
   if state['capture']:
    cache[k,'raw']=raw[k].double().clone();cache[k,'mlp']=out[:,-1].clone()
    x=cache[k,'raw'];w=weights[k];formula_err=max(formula_err,float((normalized_mlp(x,w)-out[:,-1]).abs().max()));formula_rel=max(formula_rel,float((normalized_mlp(x,w)-out[:,-1]).norm()/out[:,-1].norm()))
    if k not in checked:
     checked.add(k)
     with torch.inference_mode(False),torch.enable_grad():
      xx=x[:1].clone();v=torch.cos(torch.arange(xx.numel(),device=xx.device,dtype=torch.double)).reshape_as(xx);v=v*xx.norm()/v.norm();ww=tuple(t.clone() for t in w);manual=jvp(xx,v,ww);_,auto=torch.autograd.functional.jvp(lambda z:normalized_mlp(z,ww),xx,v);fd=(normalized_mlp(xx+1e-4*v,ww)-normalized_mlp(xx-1e-4*v,ww))/2e-4;jvp_err=max(jvp_err,float((manual-auto).norm()/auto.norm()));fd_err=max(fd_err,float((manual-fd).norm()/manual.norm()))
    return
   if state['mode']=='live':return
   y=out.clone();replacement=cache[k,'mlp']
   replacement=replacement+jvp(donor[k],raw[k].double()-cache[k,'raw'],weights[k]).to(replacement)
   y[:,-1]=replacement;return y
  return hook
 checked=set()
 for k,b in enumerate(model.blocks):
  if k>=8:
   handles.append(b.register_forward_pre_hook(prehook(k)));handles.append(b.attn.register_forward_hook(atthook(k)));handles.append(b.mlp.register_forward_hook(mlphook(k)))
 def finalhook(mod,args,out):stash['final']=out[0][:,-1].double().cpu()
 handles.append(model.blocks[-1].register_forward_hook(finalhook));buckets=defaultdict(list)
 for i,e in enumerate(ex):buckets[len(e['ids'])].append(i)
 records=[];saved={};actual=0;identity=0.;replay=0.;skiperr=0.;skiprel=0.;readouterr=0.;selferr=0.;nativecache={};donor={};donor_indices=[]
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),12):
    ix=ii[st:st+12];tokens=torch.tensor([ex[i]['ids'] for i in ix],device='cuda');state['capture']=True;final_logits(model,tokens);actual+=len(ix)
    for key,v in cache.items():
     for n,i in enumerate(ix):nativecache[i,key]=v[n].clone()
  state['capture']=False
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),12):
    ix=ii[st:st+12];tokens=torch.tensor([ex[i]['ids'] for i in ix],device='cuda')
    cache={key:torch.stack([nativecache[i,key] for i in ix]) for key in cache}
    att=model.blocks[8].attn;original=att.squared_attention;db=[];amps=[]
    for i in ix:
     e=ex[i];j=index[e['id'],1-e['order']];p=e['source_positions'][0];db.append((data['bus'][j][p,3]-data['bus'][i][p,3]).double());amps.append(previous[e['id'],e['order'],8,3,'common_positive']['routing_coefficient'])
    delta=torch.stack(db).cuda()*torch.tensor(amps,device='cuda',dtype=torch.double)[:,None];projected=delta.cpu()@att.c_proj.weight[:,384:512].double().cpu().T
    for mode,scale in [(m,s) for m in MODES for s in SCALES]:
     state['mode']=mode
     jj=[]
     for i in ix:
      e=ex[i];family=e['family'] if mode in ['live','linearized'] else ('copy' if mode=='both_copy' else 'capital' if mode=='both_capital' else 'capital' if e['family']=='copy' else 'copy');j=matched[e['pair'],e['format'],e['order'],family];jj.append(j);donor_indices.append({'recipient':i,'donor':j,'mode':mode,'scale':scale})
     donor={k:torch.stack([nativecache[j,(k,'raw')] for j in jj]) for k in range(8,18)}
     for sign in [0,1,-1]:
      def edit(q,k,v,q2,k2):
       y=original(q,k,v,q2,k2)
       if sign:y=y.clone();y[:,3,-1]+=(sign*scale*delta).to(y)
       return y
      att.squared_attention=edit
      try:logits=final_logits(model,tokens)
      finally:att.squared_attention=original
      actual+=len(ix)
      for n,i in enumerate(ix):
       e=ex[i];z=stash['final'][n];saved[i,mode,scale,sign]=z;r=(z.square().mean()+eps).sqrt()
       if sign==0:identity=max(identity,float((z-prior[i,8,3,0]).abs().max()))
       if mode=='linearized':selferr=max(selferr,float((z-selfprior[i,mode,scale,sign]).abs().max()))
       if mode=='live' and scale==1:replay=max(replay,float((z-prior[i,8,3,sign]).abs().max()))
       for output,ids in idsby[i].items():
        w=U[ids];b=torch.zeros(2,dtype=torch.double) if bias is None else bias[ids]
        for norm in ['native','shared_fixed','shared_fixed_no_cap']:
         y=w@(z/(r if norm=='native' else shared[i]))+b
         if norm!='shared_fixed_no_cap':y=30*torch.tanh(y/30)
         if norm=='native':readouterr=max(readouterr,float((y-logits[n].double().cpu()[ids]).abs().max()))
         records.append({key:e[key] for key in ['pair','format','family','order','id']}|{'mode':mode,'scale':scale,'sign':sign,'amplitude':scale*amps[n],'output':output,'normalization':norm,'margin':float(y[0]-y[1])})
   print('DONE',length,actual,flush=True)
 finally:
  for handle in handles:handle.remove()
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_finalstate_max_abs':identity,'self_jacobian_finalstate_replay':selferr,'live_finalstate_replay_max_abs':replay,'analytic_jvp_vs_autograd_relative':jvp_err,'analytic_jvp_vs_finite_difference_relative':fd_err,'double_formula_vs_native_max_abs':formula_err,'double_formula_vs_native_relative_norm':formula_rel,'jacobian_layers_checked':len(checked),'selected_readout_max_abs':readouterr,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['margin']) for r in records)}
 assert audits['hashes_unchanged'] and identity<1e-4 and selferr<1e-4 and replay<1e-4 and jvp_err<1e-8 and fd_err<1e-6 and len(checked)==10 and formula_rel<1e-5 and readouterr<1e-4 and actual==5952 and audits['finite'],audits
 torch.save(saved,ART/f'{S}_final_states.pt');torch.save({(i,k):v.cpu() for (i,k),v in nativecache.items()},ART/f'{S}_native_mlp_states.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'donor_indices':donor_indices,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

"""H878 native-update freezes with live unselected downstream computation."""
from run_bilin18_final_scale_clamp import *
S='bilin18_downstream_update_freeze'
MODES={'live':([],[]),'mlp17':([17],[]),'mlp15_17':(list(range(15,18)),[]),'mlp8_17':(list(range(8,18)),[]),'attention9_17':([],list(range(9,18))),'joint':(list(range(8,18)),list(range(9,18)))}
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'panel':ART/'bilin18_reader_recombination_panel_captures.pt','screen':ART/'bilin18_reader_recombination_panel.json','live':ART/'bilin18_reader_gain_confirmation.json','finals':ART/'bilin18_final_scale_clamp_final_states.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H878','hashes':hashes,'modes':MODES,'design':'L8H3 common symmetric writes; freeze selected final-token updates to recipient unedited native values; attention L8 remains live to retain injected write. All 192 H873 contexts, all identity arms. Native/shared fixed/shared fixed uncapped readouts. Posthoc diagnostic, no reserved test.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];prior=torch.load(paths['finals'],weights_only=False);live=json.loads(paths['live'].read_text());previous={(r['id'],r['order'],r['layer'],r['head'],r['mode']):r for r in live['records']};screen=json.loads(paths['screen'].read_text());tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 idsby={i:{typ:[continuation_id(tok,e['text'],' '+v[part]) for v in screen['config']['pairs'][e['pair']]] for typ,part in [('country',0),('capital',1)]} for i,e in enumerate(ex)};index={(e['id'],e['order']):i for i,e in enumerate(ex)};matched={(e['pair'],e['format'],e['order'],e['family']):i for i,e in enumerate(ex)};eps=torch.finfo(torch.float32).eps;radii=torch.stack([(prior[i,8,3,0].square().mean()+eps).sqrt() for i in range(len(ex))]);shared=torch.stack([(radii[i]*radii[matched[e['pair'],e['format'],e['order'],'capital' if e['family']=='copy' else 'copy']]).sqrt() for i,e in enumerate(ex)])
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 U=model.lm_head.weight.double().cpu();bias=None if model.lm_head.bias is None else model.lm_head.bias.double().cpu();skip=float(np.prod([float(b.lambdas[0]) for b in model.blocks[9:]]));cache={};state={'capture':True,'mode':'live'};handles=[];stash={}
 def makehook(k,typ):
  def hook(mod,args,out):
   y=out[0] if typ=='attention' else out
   if state['capture']:cache[k,typ]=y[:,-1].clone();return
   selected=MODES[state['mode']][1 if typ=='attention' else 0]
   if k in selected:
    z=y.clone();z[:,-1]=cache[k,typ]
    return (z,*out[1:]) if typ=='attention' else z
  return hook
 for k,b in enumerate(model.blocks):
  handles.append(b.attn.register_forward_hook(makehook(k,'attention')));handles.append(b.mlp.register_forward_hook(makehook(k,'mlp')))
 def finalhook(mod,args,out):stash['final']=out[0][:,-1].double().cpu()
 handles.append(model.blocks[-1].register_forward_hook(finalhook));buckets=defaultdict(list)
 for i,e in enumerate(ex):buckets[len(e['ids'])].append(i)
 records=[];saved={};actual=0;identity=0.;replay=0.;skiperr=0.;skiprel=0.;readouterr=0.
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),12):
    ix=ii[st:st+12];tokens=torch.tensor([ex[i]['ids'] for i in ix],device='cuda');state['capture']=True;final_logits(model,tokens);actual+=len(ix);state['capture']=False
    att=model.blocks[8].attn;original=att.squared_attention;db=[];amps=[]
    for i in ix:
     e=ex[i];j=index[e['id'],1-e['order']];p=e['source_positions'][0];db.append((data['bus'][j][p,3]-data['bus'][i][p,3]).double());amps.append(previous[e['id'],e['order'],8,3,'common_positive']['routing_coefficient'])
    delta=torch.stack(db).cuda()*torch.tensor(amps,device='cuda',dtype=torch.double)[:,None];projected=delta.cpu()@att.c_proj.weight[:,384:512].double().cpu().T
    for mode in MODES:
     state['mode']=mode
     for sign in [0,1,-1]:
      def edit(q,k,v,q2,k2):
       y=original(q,k,v,q2,k2)
       if sign:y=y.clone();y[:,3,-1]+=(sign*delta).to(y)
       return y
      att.squared_attention=edit
      try:logits=final_logits(model,tokens)
      finally:att.squared_attention=original
      actual+=len(ix)
      for n,i in enumerate(ix):
       e=ex[i];z=stash['final'][n];saved[i,mode,sign]=z;r=(z.square().mean()+eps).sqrt()
       if sign==0:identity=max(identity,float((z-prior[i,8,3,0]).abs().max()))
       if mode=='live':replay=max(replay,float((z-prior[i,8,3,sign]).abs().max()))
       if mode=='joint' and sign:
        change=z-prior[i,8,3,0];pred=sign*skip*projected[n];skiperr=max(skiperr,float((change-pred).abs().max()));skiprel=max(skiprel,float((change-pred).norm()/pred.norm()))
       for output,ids in idsby[i].items():
        w=U[ids];b=torch.zeros(2,dtype=torch.double) if bias is None else bias[ids]
        for norm in ['native','shared_fixed','shared_fixed_no_cap']:
         y=w@(z/(r if norm=='native' else shared[i]))+b
         if norm!='shared_fixed_no_cap':y=30*torch.tanh(y/30)
         if norm=='native':readouterr=max(readouterr,float((y-logits[n].double().cpu()[ids]).abs().max()))
         records.append({key:e[key] for key in ['pair','format','family','order','id']}|{'mode':mode,'sign':sign,'amplitude':amps[n],'output':output,'normalization':norm,'margin':float(y[0]-y[1])})
   print('DONE',length,actual,flush=True)
 finally:
  for handle in handles:handle.remove()
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_finalstate_max_abs':identity,'live_finalstate_replay_max_abs':replay,'joint_skip_max_abs':skiperr,'joint_skip_max_relative_norm':skiprel,'selected_readout_max_abs':readouterr,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['margin']) for r in records)}
 assert audits['hashes_unchanged'] and identity<1e-4 and replay<1e-4 and skiprel<1e-3 and readouterr<1e-4 and actual==3648 and audits['finite'],audits
 torch.save(saved,ART/f'{S}_final_states.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

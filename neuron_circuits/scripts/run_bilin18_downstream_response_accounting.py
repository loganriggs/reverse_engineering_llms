"""H877 exact residual-update accounting; accounting is not causal mediation."""
from run_bilin18_final_scale_clamp import *
S='bilin18_downstream_response_accounting'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'panel':ART/'bilin18_reader_recombination_panel_captures.pt','screen':ART/'bilin18_reader_recombination_panel.json','live':ART/'bilin18_reader_gain_confirmation.json','finals':ART/'bilin18_final_scale_clamp_final_states.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()}
 cfg={'experiment':'H877','hashes':hashes,'design':'Exact final-token residual update accounting for H876 common +/- writes. All 192 contexts, three frozen readers. Shared fixed RMS and uncapped readout give linear output accounting. No refit; no reserved test. Accounting is not unique causal mediation.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];prior=torch.load(paths['finals'],weights_only=False);live=json.loads(paths['live'].read_text());previous={(r['id'],r['order'],r['layer'],r['head'],r['mode']):r for r in live['records']};screen=json.loads(paths['screen'].read_text());tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 idsby={i:{typ:[continuation_id(tok,e['text'],' '+v[part]) for v in screen['config']['pairs'][e['pair']]] for typ,part in [('country',0),('capital',1)]} for i,e in enumerate(ex)}
 index={(e['id'],e['order']):i for i,e in enumerate(ex)};matched={(e['pair'],e['format'],e['order'],e['family']):i for i,e in enumerate(ex)};eps=torch.finfo(torch.float32).eps
 radii=torch.stack([(prior[i,13,0,0].square().mean()+eps).sqrt() for i in range(len(ex))]);shared=torch.stack([(radii[i]*radii[matched[e['pair'],e['format'],e['order'],'capital' if e['family']=='copy' else 'copy']]).sqrt() for i,e in enumerate(ex)])
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 U=model.lm_head.weight.double().cpu();weights=[float(np.prod([float(b.lambdas[0]) for b in model.blocks[k+1:]])) for k in range(18)];stash={};handles=[]
 def makehook(k,typ):
  def hook(mod,args,out):stash[k,typ]=(out[0] if typ=='attention' else out)[:,-1].double().cpu()
  return hook
 for k,b in enumerate(model.blocks):
  handles.append(b.attn.register_forward_hook(makehook(k,'attention')));handles.append(b.mlp.register_forward_hook(makehook(k,'mlp')))
 def finalhook(mod,args,out):stash['final']=out[0][:,-1].double().cpu()
 handles.append(model.blocks[-1].register_forward_hook(finalhook));buckets=defaultdict(list)
 for i,e in enumerate(ex):buckets[len(e['ids'])].append(i)
 records=[];saved={};actual=0;replay=0.;vecerr=0.;vecrel=0.;outerr=0.;directerr=0.;directrel=0.
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),12):
    ix=ii[st:st+12];tokens=torch.tensor([ex[i]['ids'] for i in ix],device='cuda')
    for l,h in TARGETS:
     att=model.blocks[l].attn;original=att.squared_attention;db=[];amps=[]
     for i in ix:
      e=ex[i];j=index[e['id'],1-e['order']];p=e['source_positions'][0];db.append((data['bus'][j][p,h]-data['bus'][i][p,h]).double());amps.append(previous[e['id'],e['order'],l,h,'common_positive']['routing_coefficient'])
     delta=torch.stack(db).cuda()*torch.tensor(amps,device='cuda',dtype=torch.double)[:,None];arms={}
     for sign in [0,1,-1]:
      def edit(q,k,v,q2,k2):
       y=original(q,k,v,q2,k2)
       if sign:y=y.clone();y[:,h,-1]+=(sign*delta).to(y)
       return y
      att.squared_attention=edit
      try:final_logits(model,tokens)
      finally:att.squared_attention=original
      actual+=len(ix);arms[sign]={key:val.clone() for key,val in stash.items()}
      for n,i in enumerate(ix):replay=max(replay,float((stash['final'][n]-prior[i,l,h,sign]).abs().max()))
     for n,i in enumerate(ix):
      e=ex[i];amp=amps[n];components={f'L{k}_{typ}':weights[k]*(arms[1][k,typ][n]-arms[-1][k,typ][n])/(2*amp) for k in range(l,18) for typ in ['attention','mlp']};total=(arms[1]['final'][n]-arms[-1]['final'][n])/(2*amp);summed=sum(components.values());vecerr=max(vecerr,float((summed-total).abs().max()));vecrel=max(vecrel,float((summed-total).norm()/total.norm()))
      direct=weights[l]*(model.blocks[l].attn.c_proj.weight[:,128*h:128*(h+1)].double().cpu()@db[n]);directerr=max(directerr,float((components[f'L{l}_attention']-direct).abs().max()));directrel=max(directrel,float((components[f'L{l}_attention']-direct).norm()/direct.norm()));saved[i,l,h]={'components':components,'total':total}
      for typ,ids in idsby[i].items():
       w=(U[ids[0]]-U[ids[1]])/shared[i];values={name:float(w@v) for name,v in components.items()};out=float(w@total);outerr=max(outerr,abs(sum(values.values())-out));records.append({key:e[key] for key in ['pair','format','family','order','id']}|{'layer':l,'head':h,'output':typ,'amplitude':amp,'total':out,'components':values})
   print('DONE',length,actual,flush=True)
 finally:
  for handle in handles:handle.remove()
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'finalstate_replay_max_abs':replay,'vector_recomposition_max_abs':vecerr,'vector_recomposition_max_relative_norm':vecrel,'output_recomposition_max_abs':outerr,'direct_write_max_abs':directerr,'direct_write_max_relative_norm':directrel,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['total']) and all(np.isfinite(v) for v in r['components'].values()) for r in records)}
 assert audits['hashes_unchanged'] and replay<1e-4 and vecrel<1e-3 and outerr<1e-3 and directrel<1e-4 and actual==1728 and audits['finite'],audits
 torch.save(saved,ART/f'{S}_vectors.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

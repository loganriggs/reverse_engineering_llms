"""H813 internal lexical holdout for shared versus task-specific response subspaces."""
import json,time,math
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
S='bilin18_response_subspace_controls'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 stems={'copy':'bilin18_copy_small_dose_response','ioi':'bilin18_indirect_object_response_cross'};paths={t:ART/f'{s}_captures.pt' for t,s in stems.items()};paths.update(runner=Path(__file__),model=SNAPSHOT/'pytorch_model.bin',bird=ART/'bilin18_fresh_bird_capture_captures.pt',ioi_native=ART/'bilin18_indirect_object_competence_captures.pt');hashes={k:sha(p) for k,p in paths.items()};data={t:torch.load(paths[t],weights_only=False) for t in stems};amps={'copy':[.05,.1],'ioi':[.25,.5]};ranks=[8];splits={};examples={};samples=[];lookup={}
 for task,c in data.items():
  ee={k:e for k,e in c['examples'].items() if e['format']==0};ids=sorted({e['item'] for e in ee.values()});hold=[i for n,i in enumerate(ids) if n%3==0];splits[task]={'heldout':hold,'train':[i for i in ids if i not in hold]}
  for k,e in ee.items():
   examples[task,k]=e
   if task=='copy':lookup[task,e['item'],e['query'],e['same_key']]=k
   else:lookup[task,e['item'],e['order'],e['query']]=k
  for key,v in c['captures'].items():
   rid=key[0];amp=key[-1]
   if rid not in ee or amp not in amps[task]:continue
   basekey=(*key[:-1],0.);b=c['captures'][basekey]['premlp'].double();dx=v['premlp'].double()-b;samples.append({'task':task,'id':rid,'key':key,'basekey':basekey,'item':ee[rid]['item'],'amp':amp,'state':key[1] if task=='copy' else None,'dx':dx,'heldout':ee[rid]['item'] in hold})
 cfg={'experiment':'H813','hashes':hashes,'split':splits,'ranks':ranks,'amplitudes':amps,'format':0,'fit':'Uncentered SVD of training raw L6 pre-MLP responses. Normalize each training response to unit norm, then normalize each task matrix Frobenius norm before shared fit; task-specific fit uses its own normalized rows. Orthogonal projections; no intercept.','selection':'Every third sorted lexical item held out, all variants grouped. Previously examined panels; internal fit holdout, not fresh evidence.','arms':['own','input_swap'],'controls':['full response','zero response','other-task-only rank8','Haar random rank8 seeds8130,8131,8132'],'comparisons':'Shared rank r vs specific rank r matches active dimension but specific total storage is2r; shared rank2r vs specific rankr matches total storage. No trained sparse gates, taxonomy, or computational DAG.','primary':'Heldout causal response fidelity and context-gap preservation, plus input reconstruction. Full source perturbation/recipient baseline retained.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 matrices={}
 for task in stems:
  x=torch.stack([s['dx'] for s in samples if s['task']==task and not s['heldout']]);assert x.shape[0]>16;x=x/x.norm(dim=1,keepdim=True);matrices[task]=x/x.norm()
 bases={};singular={}
 for name,x in list(matrices.items())+[('shared',torch.cat(list(matrices.values())))]:
  _,sv,vh=torch.linalg.svd(x,full_matrices=False);bases[name]=vh[:16].T.contiguous();singular[name]=sv.tolist()
 for seed in [8130,8131,8132]:
  g=torch.Generator().manual_seed(seed);bases[f'random{seed}']=torch.linalg.qr(torch.randn(1152,8,dtype=torch.float64,generator=g),mode='reduced').Q
 ortho=max(float((b.T@b-torch.eye(b.shape[1],dtype=b.dtype)).abs().max()) for b in bases.values());assert ortho<1e-10
 held=[s for s in samples if s['heldout']];needed={(s['task'],s['id']) for s in held};model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 nativecaps={'copy':torch.load(paths['bird'],weights_only=False),'ioi':torch.load(paths['ioi_native'],weights_only=False)};W=model.blocks[2].attn.c_proj.weight[:,640:768].cpu().double();directions={}
 for task,rid in needed:
  v=nativecaps[task][rid]['factors'][4] if task=='copy' else nativecaps[task][rid]['value'];directions[task,rid]=W@v.double()
 saved={};local={};native={};buckets=defaultdict(list);forwards=0
 for k in needed:buckets[len(examples[k]['ids'])].append(k)
 def raw2(m,args):saved['raw2']=m.lambdas[0]*args[0]+m.lambdas[1]*args[2]
 def capturelocal(m,args,out):saved['local']=torch.stack([(saved['raw2']+out[0])[i,examples[k]['positions'][1]] for i,k in enumerate(bb)]).cpu()
 hs=[model.blocks[2].register_forward_pre_hook(raw2),model.blocks[2].attn.register_forward_hook(capturelocal)]
 try:
  for kk in buckets.values():
   for lo in range(0,len(kk),12):
    bb=kk[lo:lo+12];z,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'));forwards+=len(bb)
    for i,k in enumerate(bb):local[k]=saved['local'][i];native[k]=z[i].cpu()
 finally:
  for h in hs:h.remove()
 mlp=model.blocks[6].mlp;assert not mlp.config.gated;L,R,D=[getattr(mlp,n).weight.double() for n in ['Left','Right','Down']]
 def response(b,dx):
  u=F.rms_norm(b.float().cuda(),(1152,)).double();v=F.rms_norm((b+dx).float().cuda(),(1152,)).double();return (F.linear(F.linear(v,L)*F.linear(v,R)-F.linear(u,L)*F.linear(u,R),D)).cpu()
 jobs=[];targets={};geometry=[];directionerr=[]
 for si,s in enumerate(held):
  task=s['task'];rid=s['id'];e=examples[task,rid];c=data[task]['captures'];other=lookup[task,e['item'],e['query'],not e['same_key']] if task=='copy' else lookup[task,e['item'],e['order'],1-e['query']];directionerr.append(float((directions[task,rid]-directions[task,other]).abs().max()));b=c[s['basekey']]['premlp'].double()
  for arm in ['own','input_swap']:
   dk=s['key'] if arm=='own' else (other,*s['key'][1:]);dx=c[dk]['premlp'].double()-c[(*dk[:-1],0.)]['premlp'].double()
   for name,rank in [('full',1152),('zero',0)]+[(n,r) for n in ['shared','specific','cross_task','random8130','random8131','random8132'] for r in ranks]:
    if name=='full':proj=dx
    elif name=='zero':proj=torch.zeros_like(dx)
    else:
     basisname='shared' if name=='shared' else task if name=='specific' else ('ioi' if task=='copy' else 'copy') if name=='cross_task' else name;B=bases[basisname][:,:rank];proj=B@(B.T@dx)
    tk=si,arm,name,rank;targets[tk]=response(b,proj);geometry.append({'sample':si,'task':task,'id':rid,'item':s['item'],'amp':s['amp'],'arm':arm,'dictionary':name,'rank':rank,'relative_input_error':float((dx-proj).norm()/dx.norm())});jobs.append({'s':s,'target':tk,'arm':arm,'dictionary':name,'rank':rank})
 # Baseline runs included explicitly, reusing original MLP baseline and zero perturbation.
 for si,s in enumerate(held):jobs.append({'s':s,'target':None,'arm':'baseline','dictionary':'baseline','rank':0})
 buckets=defaultdict(list)
 for j in jobs:buckets[len(examples[j['s']['task'],j['s']['id']]['ids'])].append(j)
 outputs={};baseline={};records=[]
 def inject(m,args,out):
  y=out[0].clone()
  for i,j in enumerate(bb):
   s=j['s'];task=s['task'];rid=s['id'];e=examples[task,rid];p=e['positions'][1];amp=0 if j['arm']=='baseline' else s['amp']
   if task=='copy':don=lookup[task,e['item'],e['query'],s['state']];desired=(local[task,don].double()+amp*directions[task,rid]).to(y);y[i,p]=desired-saved['raw2'][i,p]
   else:y[i,p]+=amp*directions[task,rid].to(y)
  return y,out[1]
 def patch(m,args,out):
  y=out.clone()
  for i,j in enumerate(bb):
   s=j['s'];b=data[s['task']]['captures'][s['basekey']]['mlpout'];d=targets[j['target']] if j['target'] is not None else torch.zeros_like(b);y[i,-1]=(b.double()+d).to(y)
  return y
 for jj in buckets.values():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];hs=[model.blocks[2].register_forward_pre_hook(raw2),model.blocks[2].attn.register_forward_hook(inject),model.blocks[6].mlp.register_forward_hook(patch)]
   try:z,_=run(model,torch.tensor([examples[j['s']['task'],j['s']['id']]['ids'] for j in bb],device='cuda'))
   finally:
    for h in hs:h.remove()
   forwards+=len(bb)
   for i,j in enumerate(bb):
    s=j['s'];sk=s['task'],s['key']
    if j['arm']=='baseline':baseline[sk]=z[i].cpu()
    else:outputs[sk,j['arm'],j['dictionary'],j['rank']]=z[i].cpu()
 old={t:json.loads((ART/f'{stems[t]}.json').read_text()) for t in stems};replay=[]
 for (sk,arm,name,rank),z in outputs.items():
  task,key=sk;rid=key[0];e=examples[task,rid];base=baseline[sk].double();p=z.double().log_softmax(0);p0=base.log_softmax(0);token=e['target'] if task=='copy' else e['candidate_ids'][0];target=e['target'] if task=='copy' else e['candidate_ids'][e['query_display_index']];val=float(p[token]-p0[token]);records.append({'task':task,'id':rid,'item':e['item'],'amp':key[-1],'state':key[1] if task=='copy' else None,'context':not e['same_key'] if task=='copy' else e['query_display_index']==0,'arm':arm,'dictionary':name,'rank':rank,'logprob_response':val,'margin_response':None if task=='copy' else float((z[e['candidate_ids'][0]]-z[e['candidate_ids'][1]])-(base[e['candidate_ids'][0]]-base[e['candidate_ids'][1]])),'correct':int(z.argmax())==target})
  if name=='full':
   rr=[r for r in old[task]['records'] if r['id']==rid and r['delta']==key[-1] and r['arm']==arm and (task!='copy' or r['state']==key[1])];assert len(rr)==1;replay.append(abs(val-rr[0]['response' if task=='copy' else 'first_logprob_response']))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'basis_orthogonality_max_abs':ortho,'paired_direction_max_abs':max(directionerr),'full_response_historical_replay_max_abs':max(replay),'train_holdout_items_disjoint':all(not(set(x['train'])&set(x['heldout'])) for x in splits.values()),'finite':all(math.isfinite(r['logprob_response']) for r in records),'whole_model_forwards':forwards,'test_evaluations':0};print(audits);assert audits['hashes_unchanged'] and audits['train_holdout_items_disjoint'] and audits['finite'] and max(replay)<1e-4 and max(directionerr)==0
 torch.save({'bases':bases,'singular_values':singular,'split':splits},ART/f'{S}_bases.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'geometry':geometry,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
if __name__=='__main__':main()

"""H838 prespecified full-distribution validation of frozen state featurizers."""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_early_local_geometry import sha
S='bilin18_featurizer_downstream_confirmation'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_frozen_featurizer_confirmation_captures.pt','lock':ROOT/'configs/bilin18_frozen_featurizer_confirmation.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};d=torch.load(paths['data'],weights_only=False);lock=json.loads(paths['lock'].read_text())['downstream_lock'];docs=lock['document_indices'];probes=lock['probe_indices'];amps=lock['signed_doses'];assert docs==list(range(32)) and probes==[0,8] and amps==[-.1,.1];names=[n for l,n in d['predicted_states'] if l==1];cfg={'experiment':'H838','hashes':hashes,'locked_subset':lock,'models':names,'reference':'Natural finite response: inject delta=amp*recipient preMLP RMS*unit probe into final attention output, before MLP. Own analytic and approximated responses replace final MLP output with recipient native output +amp*derivative; skip perturbation stays. Zero removes only module response.','primary':'KL(natural finite intervention distribution || approximate intervention distribution), both signs and both probes; normalize aggregate against zero response.','replay':'Native states recaptured at exact prefix/batch precision, zero-amplitude replacement vs native, normalized target-input replay. No outcome selection.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 buckets=defaultdict(list)
 for i in docs:buckets[d['positions'][i]].append(i)
 native={};states={};mlpbase={};replay=[];forwards=0
 for pos,ii in buckets.items():
  for st in range(0,len(ii),8):
   bb=ii[st:st+8];raw={};capt={};hs=[]
   for l in [1,2]:
    def bp(m,args,l=l):raw[l]=(m.lambdas[0]*args[0]+m.lambdas[1]*args[2])[:,-1]
    def ah(m,args,out,l=l):capt[l,'b']=(raw[l]+out[0][:,-1]).cpu()
    def mh(m,args,out,l=l):capt[l,'m']=out[:,-1].cpu()
    hs.extend([model.blocks[l].register_forward_pre_hook(bp),model.blocks[l].attn.register_forward_hook(ah),model.blocks[l].mlp.register_forward_hook(mh)])
   try:z,_=run(model,torch.stack([d['all_token_ids'][i,:pos+1] for i in bb]).cuda())
   finally:
    for h in hs:h.remove()
   for n,i in enumerate(bb):
    native[i]=z[n].cpu()
    for l in [1,2]:states[l,i]=capt[l,'b'][n];mlpbase[l,i]=capt[l,'m'][n];truth=d['captures'][l]['premlp'][i];replay.append(float((states[l,i]-truth).norm()/truth.norm()))
   forwards+=len(bb)
 targets={};deltas={};jobs=defaultdict(list);eps=torch.finfo(torch.float32).eps
 for l in [1,2]:
  mlp=model.blocks[l].mlp;assert not mlp.config.gated;L,R,D=[getattr(mlp,n).weight.double() for n in ['Left','Right','Down']]
  def response(b,v):
   r=(b.square().mean()+eps).sqrt();u=b/r;q=v/r-b*(b*v).mean()/r.pow(3);return F.linear(F.linear(u,L)*F.linear(q,R)+F.linear(q,L)*F.linear(u,R),D).cpu()
  for i in docs:
   b=states[l,i].double().cuda();radius=(b.square().mean()+eps).sqrt()
   for pi in probes:
    v=d['directions'][l][pi,i].double().cuda()*radius;deltas[l,i,pi]=v.cpu();targets[l,i,pi,'own']=response(b,v)
    for name in names:targets[l,i,pi,name]=response(d['predicted_states'][l,name][i].cuda(),v)
    targets[l,i,pi,'zero']=torch.zeros(1152,dtype=torch.float64)
    for amp in amps:
     for name in ['full','own','zero']+names:jobs[l,d['positions'][i]].append((l,i,pi,amp,name))
   jobs[l,d['positions'][i]].append((l,i,0,0.,'zero'))
 outputs={};zeroerr=[];inputerr=[]
 for (l,pos),jj in jobs.items():
  for st in range(0,len(jj),8):
   bb=jj[st:st+8];delta=torch.stack([deltas[l,i,pi]*amp for _,i,pi,amp,_ in bb]);patch=[]
   for _,i,pi,amp,name in bb:patch.append(mlpbase[l,i].double()+(targets[l,i,pi,name]*amp if name!='full' else torch.zeros(1152,dtype=torch.float64)))
   patch=torch.stack(patch)
   def ah(m,args,out):
    y,bus=out;y=y.clone();y[:,-1]+=delta.to(y);return y,bus
   def pre(m,args):
    expected=F.rms_norm((torch.stack([states[l,i] for _,i,_,_,_ in bb]).double()+delta).float().cuda(),(1152,));actual=args[0][:,-1];inputerr.append(float((actual-expected).norm()/expected.norm()))
   def mh(m,args,out):
    y=out.clone()
    for n,key in enumerate(bb):
     if key[-1]!='full':y[n,-1]=patch[n].to(y)
    return y
   hs=[model.blocks[l].attn.register_forward_hook(ah),model.blocks[l].mlp.register_forward_pre_hook(pre),model.blocks[l].mlp.register_forward_hook(mh)]
   try:z,_=run(model,torch.stack([d['all_token_ids'][i,:pos+1] for _,i,_,_,_ in bb]).cuda())
   finally:
    for h in hs:h.remove()
   for n,key in enumerate(bb):
    outputs[key]=z[n].cpu()
    if key[3]==0:zeroerr.append(float((z[n].cpu()-native[key[1]]).abs().max()))
   forwards+=len(bb)
  print('DONE',l,pos,flush=True)
 records=[]
 for l in [1,2]:
  for i in docs:
   for pi in probes:
    for amp in amps:
     ref=outputs[l,i,pi,amp,'full'].double().log_softmax(0)
     for name in ['own','zero']+names:
      cand=outputs[l,i,pi,amp,name].double().log_softmax(0);records.append({'layer':l,'document':i,'probe':pi,'amp':amp,'model':name,'KL':float((ref.exp()*(ref-cand)).sum()),'TV':float((ref.exp()-cand.exp()).abs().sum()/2)})
 summary=[]
 for l in [1,2]:
  for pi in probes:
   zero=np.mean([r['KL'] for r in records if r['layer']==l and r['probe']==pi and r['model']=='zero'])
   for name in ['own','zero']+names:
    rr=[r for r in records if r['layer']==l and r['probe']==pi and r['model']==name];summary.append({'layer':l,'probe':pi,'model':name,'mean_KL':float(np.mean([r['KL'] for r in rr])),'KL_over_zero':float(np.mean([r['KL'] for r in rr])/zero)})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_preMLP_replay_max_relative':max(replay),'normalized_input_replay_max_relative':max(inputerr),'zero_amp_logit_replay_max_abs':max(zeroerr),'whole_model_forwards':forwards,'test_evaluations':0,'all_locked_documents':len(docs),'finite':all(np.isfinite(r['KL']) for r in records),'minimum_KL':min(r['KL'] for r in records)};print(audits,flush=True);assert audits['hashes_unchanged'] and max(replay)<1e-4 and max(inputerr)<1e-4 and max(zeroerr)<1e-4 and audits['finite'] and audits['minimum_KL']>=-1e-12;torch.save({'outputs':outputs,'native':native},ART/f'{S}_logits.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
if __name__=='__main__':main()

"""H982 frozen scalar channel scope, nuisance changes and diverse task controls."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from run_bilin18_single_state_composition import predict
from pathlib import Path
from collections import defaultdict
import torch,json,numpy as np,time
S='bilin18_scalar_channel_scope'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'fresh':ART/'bilin18_fresh_single_state_panel_captures.pt','fresh_gate':ART/'bilin18_fresh_single_state_panel.json','bank':ART/'bilin18_expanded_task_bank_captures.pt','bank_gate':ART/'bilin18_expanded_task_bank.json','model_params':ART/'bilin18_single_state_composition_models.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};fresh=torch.load(paths['fresh'],weights_only=False);bank=torch.load(paths['bank'],weights_only=False);fg=json.loads(paths['fresh_gate'].read_text());bg=json.loads(paths['bank_gate'].read_text());saved=torch.load(paths['model_params'],weights_only=False);params=saved['params']['aware_n4'];u=saved['output_direction'];examples={};native={};writes={};inputs={};positions={};edges=[]
 for i,e in enumerate(fresh['examples']):
  k=f'f{i}';examples[k]=e;native[k]=fresh['native'][i];writes[k]=fresh['captures'][i]['adjective_write'];inputs[k]=fresh['captures'][i]['normalized'];positions[k]=e['adjective_position']
 qmap={(q['format'],q['source']):q for q in fresh['quartets']};gates={(r['format'],r['cohort']):r['pass_gate'] for r in fg['summary']}
 for q in fresh['quartets']:
  if q['format'] not in ['book','story']:continue
  for rec,i in enumerate(q['indices']):
   common=dict(domain='sentiment',format=q['format'],cohort=q['cohort'],native_gate=gates[q['format'],q['cohort']],position='adjective_last_subtoken')
   for don in range(4):
    if don==rec or (fresh['examples'][i]['a']!=fresh['examples'][q['indices'][don]]['a'] and fresh['examples'][i]['n']!=fresh['examples'][q['indices'][don]]['n']):continue
    edges.append(common|dict(kind='valence_change',recipient=f'f{i}',donor=f'f{q["indices"][don]}'))
   other=qmap[q['format'],(q['source']//4)*4+(q['source']+1)%4];edges.append(common|dict(kind='same_valence_lexical',recipient=f'f{i}',donor=f'f{other["indices"][rec]}'))
   other=qmap['story' if q['format']=='book' else 'book',q['source']];edges.append(common|dict(kind='same_valence_context',recipient=f'f{i}',donor=f'f{other["indices"][rec]}'))
 eligible=['agreement','capital_recall','entity_binding','pronoun_gender','sequence_copy','spatial_inverse'];needed=set()
 for r in bg['records']:
  if r['family'] not in eligible or not r['same_final_token']:continue
  gate=next(s['pass_gate'] for s in bg['summary'] if s['family']==r['family'] and s['split']==r['split'])
  for rec,i in enumerate(r['indices']):
   di=r['indices'][1-rec];needed.update([i,di]);edges.append(dict(domain=r['family'],format=r['split'],cohort='bank',native_gate=gate,position='last_token',kind='task_change',recipient=f'b{i}',donor=f'b{di}'))
 cfg={'experiment':'H982','hashes':hashes,'design':'Frozen H979 aware8 scalar readout and output direction; fresh book/story valence-changing edges, same-valence lexical/context edges, plus six task-bank families. Compare full MLP3 donor-write edit, exact scalar projection, predicted scalar, orthogonal complement, identity, and +/- scalar-norm-matched random edits.','position':'Sentiment uses adjective last subtoken, non-sentiment uses final token. This deliberately tests position/task transfer, with a position confound; poor transfer does not isolate semantic cause.','metrics':'Task margin change and KL(native||edited) describe impact. KL(full edit||candidate) and KL(exact scalar edit||predicted) assess different fidelity scopes. Same-valence label equality does not imply true activations/distributions should be invariant.','holdouts':'Copying/spatial tasks remain training holdouts; evaluation only. No fitting, original reserved test or per-example competence filtering.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 indices=[];buf={}
 def ih(module,args):buf['input']=args[0][:,-1].detach().cpu()
 def oh(module,args,out):buf['write']=out[:,-1].detach().cpu()
 hooks=[model.blocks[3].mlp.register_forward_pre_hook(ih),model.blocks[3].mlp.register_forward_hook(oh)];actual=0;replay=0.
 try:
  for length in sorted({len(bank['examples'][i]['ids']) for i in needed}):
   ii=sorted(i for i in needed if len(bank['examples'][i]['ids'])==length)
   for st in range(0,len(ii),24):
    indices=ii[st:st+24];y,_=forward(model,torch.tensor([bank['examples'][i]['ids'] for i in indices],device='cuda'));actual+=len(indices)
    for k,i in enumerate(indices):
     key=f'b{i}';examples[key]=bank['examples'][i];native[key]=bank['native'][i];inputs[key]=buf['input'][k];writes[key]=buf['write'][k];positions[key]=len(examples[key]['ids'])-1;replay=max(replay,float((y[k].double().cpu()-native[key]).abs().max()))
 finally:
  for h in hooks:h.remove()
 pred={k:float(predict(x.double().unsqueeze(0),params)[0]) for k,x in inputs.items()};truth={k:float(w.double()@u) for k,w in writes.items()};jobs=[];normerr=0.
 for ei,e in enumerate(edges):
  i,di=e['recipient'],e['donor'];delta=writes[di]-writes[i];scalar=(truth[di]-truth[i])*u.float();ps=(pred[di]-pred[i])*u.float();gen=torch.Generator().manual_seed(982000+ei);z=torch.randn(1152,generator=gen);z=z/z.norm()*scalar.norm();normerr=max(normerr,float(abs(z.norm()-scalar.norm())/scalar.norm().clamp_min(1e-20)))
  for arm,edit in dict(full=delta,scalar=scalar,predicted=ps,orthogonal=delta-scalar,identity=delta*0,random_plus=z,random_minus=-z).items():jobs.append(dict(e=e,ei=ei,i=i,arm=arm,edit=edit))
 current=[]
 def patch(module,args,out):
  y=out.clone()
  for k,j in enumerate(current):i=j['i'];y[k,positions[i]]=writes[i].to(y)+j['edit'].to(y)
  return y
 hook=model.blocks[3].mlp.register_forward_hook(patch);refs={};records=[];identity=0.
 def execute(jobs):
  nonlocal current,actual,identity
  buckets=defaultdict(list)
  for j in jobs:buckets[len(examples[j['i']]['ids'])].append(j)
  for length,jj in sorted(buckets.items()):
   for st in range(0,len(jj),24):
    current=jj[st:st+24];y,_=forward(model,torch.tensor([examples[j['i']]['ids'] for j in current],device='cuda'));actual+=len(current)
    for k,j in enumerate(current):
     out=y[k].double().cpu();base=native[j['i']];lp=out.log_softmax(0);ln=base.log_softmax(0)
     if j['arm'] in ['full','scalar']:refs[j['ei'],j['arm']]=out
     bm=measure(base,examples[j['i']]['groups'])['margin'];margin=measure(out,examples[j['i']]['groups'])['margin'];r=j['e']|dict(edge_id=j['ei'],arm=j['arm'],margin_delta=margin-bm,KL_native=float((ln.exp()*(ln-lp)).sum()),edit_norm=float(j['edit'].norm()))
     if j['arm']=='identity':identity=max(identity,float((out-base).abs().max()))
     for name in ['full','scalar']:
      if (j['ei'],name) in refs:
       lr=refs[j['ei'],name].log_softmax(0);r['KL_'+name]=float((lr.exp()*(lr-lp)).sum());r['baseline_'+name]=float((lr.exp()*(lr-ln)).sum())
     records.append(r)
 try:
  execute([j for j in jobs if j['arm']=='full']);execute([j for j in jobs if j['arm']=='scalar']);execute([j for j in jobs if j['arm'] not in ['full','scalar']])
 finally:hook.remove()
 summary=[];grouped=defaultdict(list)
 for r in records:grouped[r['domain'],r['kind'],r['native_gate'],r['arm']].append(r)
 for key,rr in grouped.items():
  s=dict(zip(['domain','kind','native_gate','arm'],key))|dict(n=len(rr),margin_rms=float(np.mean([r['margin_delta']**2 for r in rr])**.5),mean_native_KL=float(np.mean([r['KL_native'] for r in rr])))
  for name in ['full','scalar']:
   if all('KL_'+name in r for r in rr):s[name+'_KL_ratio']=sum(r['KL_'+name] for r in rr)/max(sum(r['baseline_'+name] for r in rr),1e-20)
  summary.append(s)
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),whole_model_forwards=actual,expected_forwards=len(needed)+len(jobs),capture_logits_max_abs=replay,identity_logits_max_abs=identity,random_norm_relative=normerr,finite=all(np.isfinite(r['KL_native']) for r in records),no_training=True,test_evaluations=0)
 assert audits['hashes_unchanged'] and actual==len(needed)+len(jobs) and max(replay,identity)<1e-4 and normerr<1e-5 and audits['finite']
 (ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,summary=summary,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

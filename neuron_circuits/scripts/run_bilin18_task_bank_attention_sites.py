"""H964 discovery-only causal MLP-write sites for expanded task bank."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward as original_forward,measure
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import torch,json,numpy as np,time
S='bilin18_task_bank_attention_sites'
@torch.inference_mode()
def forward(model,ids,capture=False,patch=None):
 cache=[];hooks=[]
 for layer,block in enumerate(model.blocks):
  if capture or (patch is not None and patch[0]==layer):
   def hook(module,args,out,layer=layer):
    y,bus=out
    if capture:cache.append(y[:,-1].detach().cpu())
    if patch is not None and patch[0]==layer:
     y=y.clone();y[:,-1]=patch[1].to(y)
     return y,bus
    return out
   hooks.append(block.attn.register_forward_hook(hook))
 try:y,_=original_forward(model,ids)
 finally:
  for hook in hooks:hook.remove()
 return y,cache
@torch.inference_mode()
def manual_patch(model,ids,layer,replacement):
 x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
 for l,block in enumerate(model.blocks):
  raw=block.lambdas[0]*x+block.lambdas[1]*x0;att,bus=block.attn(F.rms_norm(raw,(1152,)),bus)
  if l==layer:att=att.clone();att[:,-1]=replacement.to(att)
  pre=raw+att;x=pre+block.mlp(F.rms_norm(pre,(1152,)))
 return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30)

@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','model_source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py'),'screen':ART/'bilin18_expanded_task_bank.json','captures':ART/'bilin18_expanded_task_bank_captures.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};screen=json.loads(paths['screen'].read_text());data=torch.load(paths['captures'],weights_only=False);ex=data['examples'];native=data['native'];eligible=sorted(r['family'] for r in screen['summary'] if r['split']=='discovery' and r['pass_gate'] and r['family'] in ['entity_binding','sequence_copy']);rows=[r for r in screen['records'] if r['family'] in eligible];excluded=[r['id'] for r in rows if not r['same_final_token']];rows=[r for r in rows if r['same_final_token']]
 cfg={'experiment':'H964','hashes':hashes,'eligible':eligible,'design':'Capture last-token attention outputs at all18 layers for eligible families. Discovery: patch donor write bidirectionally at every layer; identity, paired +/- norm-matched Gaussian controls, same-answer discovery donor where available. Select maximum mean signed donor task-margin effect, lowest-layer tie break. Freeze selection before validation. Evaluate all other cohorts at chosen site, including failed native gates as diagnostic only.','alignment':'Last-token attention output swap, same final input token required; absolute positions and prompt lengths may differ. This is a whole-write causal assay, not identification of an input subspace or natural latent intervention.','holdouts':'sequence_copy and spatial_inverse are predeclared whole-task training holdouts among eligible tasks. Their site diagnostics are permitted, but no examples from them may enter featurizer training.','controls':'Gaussian seed fixed by pair/source/layer/recipient, identical +/- direction, norm equal to donor delta. Same-answer donor chosen by intersecting intended token groups within discovery, different prompt and pair, first available index; absence disclosed. No per-example correctness filtering.','metrics':'Signed task-margin change, output KL(patch||native recipient), KL(native donor||patch) as context-confounded diagnostic. Select on donor task effect only; report corruption specificity separately.','bus_semantics':'Patch attention output y only; preserve the native v1 bus object. The current layer MLP and all downstream layers react to the edit. Compare hook implementation to independent manual path on nontrivial patches including layer0.', 'limits':'Only4 discovery pairs/family and reused development vocabulary. Whole-write localization does not establish semantic unit or general layer4 relevance.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 indices=sorted({i for r in rows for i in r['indices']});writes={};capture_replay=0.;actual=0
 for length in sorted({len(ex[i]['ids']) for i in indices}):
  ids0=[i for i in indices if len(ex[i]['ids'])==length]
  for st in range(0,len(ids0),16):
   ii=ids0[st:st+16];ids=torch.tensor([ex[i]['ids'] for i in ii],device='cuda');y,cap=forward(model,ids,capture=True);assert len(cap)==18;actual+=len(ii)
   for k,i in enumerate(ii):writes[i]=torch.stack([p[k] for p in cap]);capture_replay=max(capture_replay,float((y[k].double().cpu()-native[i]).abs().max()))
 nuisance={};alignment=[]
 for r in rows:
  for rec,i in enumerate(r['indices']):
   intended=set(ex[i]['groups'][rec]);candidates=[]
   for other in rows:
    if other['family']!=r['family'] or other['split']!='discovery' or other['id']==r['id']:continue
    for v,ind in enumerate(other['indices']):
     if ex[ind]['text']!=ex[i]['text'] and ex[ind]['ids'][-1]==ex[i]['ids'][-1] and intended&set(ex[ind]['groups'][v]):candidates.append(ind)
   nuisance[r['id'],rec]=min(candidates) if candidates else None
 records=[];ident=0.;norm_error=0.
 def execute(subset,selected=None):
  nonlocal actual,ident,norm_error
  buckets=defaultdict(list)
  for rid,r in enumerate(subset):
   layers=range(18) if selected is None else [selected[r['family']]]
   for layer in layers:
    for rec in [0,1]:
     i=r['indices'][rec];don=r['indices'][1-rec];delta=writes[don][layer]-writes[i][layer];gen=torch.Generator().manual_seed(963000+int(r['source'])*1000+(0 if r['format']=='A' else 500)+layer*10+rec);noise=torch.randn(1152,generator=gen);noise=noise/noise.norm()*delta.norm();norm_error=max(norm_error,float(abs(noise.norm()-delta.norm())/delta.norm().clamp_min(1e-20)))
     arms=[('donor',writes[don][layer]),('identity',writes[i][layer]),('random_plus',writes[i][layer]+noise),('random_minus',writes[i][layer]-noise)]
     ni=nuisance[r['id'],rec]
     if ni is not None:arms.append(('same_answer',writes[ni][layer]))
     for kind,repl in arms:buckets[layer,len(ex[i]['ids'])].append((r,rec,i,don,kind,repl,ni,float(delta.norm())))
  for (layer,length),jj in sorted(buckets.items()):
   for st in range(0,len(jj),24):
    batch=jj[st:st+24];ids=torch.tensor([ex[t[2]]['ids'] for t in batch],device='cuda');replacement=torch.stack([t[5] for t in batch]);y,_=forward(model,ids,patch=(layer,replacement));actual+=len(batch)
    for k,(r,rec,i,don,kind,repl,ni,dnorm) in enumerate(batch):
     out=y[k].double().cpu();base=native[i];margin=measure(out,ex[i]['groups'])['margin'];bm=measure(base,ex[i]['groups'])['margin'];effect=(1 if rec==1 else -1)*(margin-bm);lp=out.log_softmax(0);ln=base.log_softmax(0);ld=native[don].log_softmax(0)
     if kind=='identity':ident=max(ident,float((out-base).abs().max()))
     records.append({key:r[key] for key in ['id','family','format','source','split','whole_task_holdout']}|dict(layer=layer,recipient=rec,kind=kind,nuisance_index=ni,donor_delta_norm=dnorm,edit_norm=float((repl-writes[i][layer]).norm()),donor_directed_delta=effect,margin=margin,native_margin=bm,KL_to_native=float((lp.exp()*(lp-ln)).sum()),donor_KL_to_patch=float((ld.exp()*(ld-lp)).sum())))
   print('DONE',layer,length,len(records),actual,flush=True)
 discovery=[r for r in rows if r['split']=='discovery'];execute(discovery);selected={};selection_scores=[]
 for family in eligible:
  scores=[]
  for layer in range(18):
   rr=[r for r in records if r['family']==family and r['layer']==layer and r['kind']=='donor'];score=float(np.mean([r['donor_directed_delta'] for r in rr]));scores.append(score);selection_scores.append(dict(family=family,layer=layer,mean_donor_effect=score))
  selected[family]=int(np.argmax(scores))
 selection={'selected_layers':selected,'scores':selection_scores,'discovery_record_count':len(records),'screen_hash':hashes['screen'],'rule':'Maximum mean donor-directed margin on discovery only; no validation access.'};(ART/f'{S}_selection.json').write_text(json.dumps(selection,indent=2)+'\n');selection_hash=sha(ART/f'{S}_selection.json');execute([r for r in rows if r['split']!='discovery'],selected)
 summary=[];grouped=defaultdict(list)
 for r in records:grouped[r['family'],r['split'],r['layer'],r['kind']].append(r)
 for key,rr in grouped.items():summary.append(dict(zip(['family','split','layer','kind'],key))|dict(n=len(rr),mean_signed_effect=float(np.mean([r['donor_directed_delta'] for r in rr])),rms_effect=float(np.mean([r['donor_directed_delta']**2 for r in rr])**.5),mean_KL=float(np.mean([r['KL_to_native'] for r in rr])),mean_edit_norm=float(np.mean([r['edit_norm'] for r in rr]))))
 manual_error=0.;audit_row=discovery[0];i,don=audit_row['indices'];ids=torch.tensor([ex[i]['ids']],device='cuda')
 for layer in [0,8,17]:
  replacement=writes[don][layer][None];one,_=forward(model,ids,patch=(layer,replacement));two=manual_patch(model,ids,layer,replacement);manual_error=max(manual_error,float((one-two).abs().max()));actual+=2
 expected=6+len(indices)+sum((18 if r['split']=='discovery' else 1)*sum(4+(nuisance[r['id'],rec] is not None) for rec in [0,1]) for r in rows)
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),selection_unchanged=sha(ART/f'{S}_selection.json')==selection_hash,capture_logits_max_abs=capture_replay,identity_logits_max_abs=ident,norm_match_max_relative_error=norm_error,whole_model_forwards=actual,expected_forwards=expected,finite=all(np.isfinite(r['KL_to_native']) and np.isfinite(r['donor_directed_delta']) for r in records),test_evaluations=0,no_featurizer_training=True,independent_hook_vs_manual_nontrivial_patch_max_abs=manual_error)
 assert audits['hashes_unchanged'] and audits['selection_unchanged'] and max(capture_replay,ident,manual_error)<1e-4 and norm_error<1e-5 and actual==expected and audits['finite']
 torch.save(dict(writes=writes,example_indices=indices,selected_layers=selected),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,summary=summary,selection=selection,competence=screen['summary'],excluded_alignment=excluded,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(selected,audits,flush=True)
if __name__=='__main__':main()

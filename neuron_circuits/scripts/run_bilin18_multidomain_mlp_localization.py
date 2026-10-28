"""H906 native donor MLP localization, all prespecified layers retained."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_multidomain_mlp_localization'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','panel':ART/'bilin18_multidomain_matched_competence_captures.pt','mapping':ART/'bilin18_partner_subspace_fidelity.json','states':ART/'bilin18_partner_subspace_fidelity_native_mlp_states.pt','previous':ART/'bilin18_paired_subspace_causal.json'};hashes={k:sha(p) for k,p in paths.items()};cfg=dict(experiment='H906',hashes=hashes,layers=list(range(8,18)),design='Same96 H905 transfer endpoints. Full native matched-opposite-source normalized-MLP-input substitution at final query, each L8–17 singly and all ten together. Fixed donor native states; recipient attention/residual/context otherwise live. No source semantics inferred from endpointorder; no layer selection. Selfnative all-layer replay control.');(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');p=torch.load(paths['panel'],weights_only=False);mapping=json.loads(paths['mapping'].read_text());ex=[p['examples'][i] for i in mapping['selection_original_indices']];states=torch.load(paths['states'],weights_only=False);uall={}
 for k in cfg['layers']:
  raw=torch.stack([states[i,(k,'raw')] for i in range(len(ex))]).double().cuda();uall[k]=raw/(raw.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
 matched={(e['domain'],e['family'],e['format'],e['pair'],e['order']):i for i,e in enumerate(ex)};evalix=[]
 for dom in ['agreement','language','binding']:
  pairs=sorted({e['pair'] for e in ex if e['domain']==dom});evalix.extend(i for i,e in enumerate(ex) if e['domain']==dom and e['pair'] in pairs[4:])
 assert len(evalix)==96;model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 weights={k:[m.weight.double() for m in [model.blocks[k].mlp.Left,model.blocks[k].mlp.Right,model.blocks[k].mlp.Down]] for k in cfg['layers']};state={'mode':'baseline'};cacheerr=0.
 def makehook(k):
  def hook(mod,args,out):
   nonlocal cacheerr
   mode=state['mode']
   if mode in ['baseline','identity'] or (mode.startswith('layer_') and int(mode[6:])!=k):return
   ix=state['indices'];u=args[0][:,-1].double();donors=ix if mode=='self_all' else [matched[ex[i]['domain'],ex[i]['family'],ex[i]['format'],ex[i]['pair'],1-ex[i]['order']] for i in ix];target=uall[k][donors];L,R,D=weights[k];change=((target@L.T)*(target@R.T)-(u@L.T)*(u@R.T))@D.T
   if mode.startswith('layer_') or mode=='self_all':cacheerr=max(cacheerr,float((u-uall[k][ix]).norm()/u.norm()))
   z=out.clone();z[:,-1]+=change.to(out);return z
  return hook
 handles=[model.blocks[k].mlp.register_forward_hook(makehook(k)) for k in cfg['layers']];modes=['identity','self_all']+['layer_'+str(k) for k in cfg['layers']]+['all'];buckets=defaultdict(list)
 for i in evalix:buckets[len(ex[i]['ids'])].append(i)
 records=[];actual=0;identity=0.;selferr=0.
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),8):
    ix=ii[st:st+8];tokens=torch.tensor([ex[i]['ids'] for i in ix],device='cuda');state.update(mode='baseline',indices=ix);base=final_logits(model,tokens).double().cpu();actual+=len(ix)
    for mode in modes:
     state['mode']=mode;y=final_logits(model,tokens).double().cpu();actual+=len(ix)
     for n,i in enumerate(ix):
      e=ex[i];ids=[g[0] for g in e['groups']];sign=1 if e['expected']==0 else -1;lp=torch.log_softmax(base[n],-1);lq=torch.log_softmax(y[n],-1)
      if mode=='identity':identity=max(identity,float((y[n]-base[n]).abs().max()))
      if mode=='self_all':selferr=max(selferr,float((y[n]-base[n]).abs().max()))
      records.append(dict(index=i,domain=e['domain'],family=e['family'],format=e['format'],pair=e['pair'],mode=mode,KL=float((lp.exp()*(lp-lq)).sum()),native_margin=sign*float(base[n,ids[0]]-base[n,ids[1]]),margin=sign*float(y[n,ids[0]]-y[n,ids[1]]),native_top1_correct=int(base[n].argmax())==ids[e['expected']],edited_top1_correct=int(y[n].argmax())==ids[e['expected']],top1_changed=int(base[n].argmax())!=int(y[n].argmax())))
   print('DONE',length,actual,flush=True)
 finally:
  for h in handles:h.remove()
 prev={r['index']:r for r in json.loads(paths['previous'].read_text())['records'] if r['mode']=='full'};replay=max(abs(r['margin']-prev[r['index']]['margin']) for r in records if r['mode']=='layer_12');audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),identity_logits_max_abs=identity,self_all_logits_max_abs=selferr,cached_activation_relative=cacheerr,L12_prior_margin_replay_max_abs=replay,whole_model_forwards=actual,expected_forwards=1344,test_evaluations=0,finite=all(np.isfinite(r['KL']) for r in records));assert audits['hashes_unchanged'] and identity<1e-4 and selferr<1e-3 and cacheerr<1e-5 and replay<1e-3 and actual==1344 and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

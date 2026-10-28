"""H720: exact additive block decomposition and live attention substitution."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch,numpy as np
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,groups as answer_groups,measure
from run_bilin18_broad_task_competence import continuation_id
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_task_transition_components'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'selection':ART/'bilin18_task_depth_transfer_summary.json','depth':ART/'bilin18_task_depth_transfer.json','depth_captures':ART/'bilin18_task_depth_transfer_captures.pt','manifest':ART/'bilin18_group_task_transfer_manifest.json','suite':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','tokenization':ROOT/'scripts/run_bilin18_broad_task_competence.py','model_source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py'),'model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};selection=json.loads(paths['selection'].read_text());selected={f:r['selected_discovery_layer'] for f,r in selection.items()};rows=json.loads(paths['manifest'].read_text());tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={}
 for row in rows:
  if row['family'] not in selected:continue
  for v,text in enumerate(row['prompts']+[row['nuisance']]):
   gs=[[continuation_id(tok,text,a) for a in g] for g in answer_groups(row['family'],row['answers'])];gs=[[a for a in g if a is not None] for g in gs];assert all(gs);examples[row['id'],v]=dict(ids=tok.encode(text,add_special_tokens=False),groups=gs,family=row['family'],layer=selected[row['family']])
 cfg=dict(experiment='H720',hashes=hashes,selected=selected,design='Exactblock: b=lambda0*xin+lambda1*x0; a=attention(rms(b),bus); m=MLP(rms(b+a)); out=b+a+m. Capture b,a,m,out for all240eligiblevariants atH719discovery-selectedsites. Output-boundary donor-minus-receiver component additions: mix,attention,MLP,attention+MLP,all. These fixed-component interventions keep nativewithinblockotherwrites. Alsolive_attention: replacedonorattention atsubmoduleoutput thenrecomputeMLPonhybridstate; preservebus/otherpositions. Naturalstrength,noamp. Fixed task/nuisance/transferpanel, no newcomponentselection.',limitations='Additivewrite decomposition notadditiveeffects orcausalmediation. Fixedattentiondelta atblockoutput differsfromchangingattentionbeforeMLP; liveattention addresses thatdifference. Mixedcarryincludesinputskip; componentscanopposeoneanother. Wholewritesnotfundamentalunits. Reusedsuite; corpusreservedtestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));print('LOADING_MODEL',flush=True);model=load_full_model('cuda');print('MODEL_LOADED',flush=True)
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 comp={};native={};buckets=defaultdict(list);forwards=0;recon=[];lambdas={}
 for k,e in examples.items():buckets[e['layer'],len(e['ids'])].append(k)
 for (layer,length),keys in buckets.items():
  block=model.blocks[layer];lambdas[str(layer)]=block.lambdas.cpu().tolist()
  for st in range(0,len(keys),12):
   bb=keys[st:st+12];cap={}
   def pre(m,args):cap['mix']=(m.lambdas[0]*args[0][:,-1]+m.lambdas[1]*args[2][:,-1]).cpu()
   def attn(m,args,out):cap['attention']=out[0][:,-1].cpu()
   def mlp(m,args,out):cap['mlp']=out[:,-1].cpu()
   def post(m,args,out):cap['out']=out[0][:,-1].cpu()
   hooks=[block.register_forward_pre_hook(pre),block.attn.register_forward_hook(attn),block.mlp.register_forward_hook(mlp),block.register_forward_hook(post)]
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:
    for h in hooks:h.remove()
   forwards+=len(bb)
   for n,k in enumerate(bb):comp[k]={name:x[n] for name,x in cap.items()};native[k]=logits[n].cpu();recon.append(float((cap['mix'][n]+cap['attention'][n]+cap['mlp'][n]-cap['out'][n]).norm()/cap['out'][n].norm()))
 assert max(recon)<1e-6;print('RECON',max(recon),flush=True);depthcap=torch.load(paths['depth_captures'],weights_only=False);nativeerr=max(float((native[k]-depthcap['native_logits'][k]).abs().max()) for k in native);del depthcap;records=[];forwards_native=forwards
 for mode in ['mix','attention','mlp','attention_mlp','all','live_attention']:
  jobs=defaultdict(list)
  for row in rows:
   if row['family'] not in selected:continue
   for rv,dv in [(0,1),(1,0),(0,2)]:jobs[selected[row['family']],len(examples[row['id'],rv]['ids'])].append(dict(id=row['id'],family=row['family'],split=row['split'],receiver=rv,donor=dv,layer=selected[row['family']],mode=mode))
  for (layer,length),jj in jobs.items():
   for st in range(0,len(jj),12):
    bb=jj[st:st+12];components=['mix','attention','mlp'] if mode=='all' else ['attention','mlp'] if mode=='attention_mlp' else ['attention'] if mode=='live_attention' else [mode];patch=torch.stack([sum(comp[r['id'],r['donor']][c]-comp[r['id'],r['receiver']][c] for c in components) for r in bb]).cuda();cap={}
    if mode=='live_attention':
     donor=torch.stack([comp[r['id'],r['donor']]['attention'] for r in bb]).cuda()
     def edit(m,args,out):
      y,bus=out;y=y.clone();y[:,-1]=donor.to(y);return y,bus
     hook=model.blocks[layer].attn.register_forward_hook(edit)
    else:
     def edit(m,args,out):
      x,bus=out;x=x.clone();x[:,-1]+=patch.to(x);return x,bus
     hook=model.blocks[layer].register_forward_hook(edit)
    def get(m,args,out):cap['out']=out[0][:,-1].cpu()
    capture=model.blocks[layer].register_forward_hook(get)
    try:logits,_=run(model,torch.tensor([examples[r['id'],r['receiver']]['ids'] for r in bb],device='cuda'))
    finally:hook.remove();capture.remove()
    forwards+=len(bb)
    for n,r in enumerate(bb):
     rk,dk=(r['id'],r['receiver']),(r['id'],r['donor']);gs=examples[rk]['groups'];base=measure(native[rk],gs);ed=measure(logits[n],gs);dn=measure(native[dk],gs);change=ed['margin']-base['margin'];sign=1 if r['donor'] in [0,2] else -1;bp=native[rk].double().log_softmax(-1);ep=logits[n].double().cpu().log_softmax(-1);records.append(dict(**r,margin_change=change,donor_directed_margin_change=sign*change,available_donor_margin=sign*(dn['margin']-base['margin']),edited_donor_correct=ed['top1'][0 if r['donor']==2 else r['donor']],edited_receiver_correct=ed['top1'][r['receiver']],receiver_kl=float((bp.exp()*(bp-ep)).sum()),injected_write_relative_norm=float(patch[n].norm().cpu()/comp[rk]['out'].norm()),resulting_output_relative_norm=float((cap['out'][n]-comp[rk]['out']).norm()/comp[rk]['out'].norm())))
  print('MODE',mode,flush=True);(ART/f'{STEM}.partial.json').write_text(json.dumps(dict(status='running',records=records),indent=2))
 depth=json.loads(paths['depth'].read_text());idx={(r['id'],r['receiver'],r['donor'],r['layer']):r for r in depth['records']};fullerr=max(abs(r['margin_change']-idx[r['id'],r['receiver'],r['donor'],r['layer']]['margin_change']) for r in records if r['mode']=='all');audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['receiver_kl']) for r in records),native_component_reconstruction_max_relative=max(recon),native_logits_previous_max_abs=nativeerr,all_components_full_swap_margin_replay_max_abs=fullerr,block_lambdas=lambdas,whole_model_forwards=forwards,native_forwards=forwards_native,test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and fullerr<1e-4;torch.save(dict(components=comp,native_logits=native),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()

"""H831 downstream effects of independently crossed MLP baseline/input responses."""
import json,time
from pathlib import Path
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
from run_bilin18_early_local_geometry import sha
S='bilin18_mlp_cross_downstream'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'grid':ART/'bilin18_mlp_state_input_cross_captures.pt','trace':ART/'bilin18_fixed_direction_depth_captures.pt','prior':ART/'bilin18_fixed_direction_recipients_captures.pt','panel':ART/'bilin18_fresh_early_geometry_panel_captures.pt','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};grid=torch.load(paths['grid'],weights_only=False);trace=torch.load(paths['trace'],weights_only=False);prior=torch.load(paths['prior'],weights_only=False);panel=torch.load(paths['panel'],weights_only=False);groups=grid['groups'];cfg={'experiment':'H831','hashes':hashes,'design':'All13 groups x4 recipients x2 fixed directions. At first downstream MLP, native recipient output +amp*analytic response; attention and skip remain natural. Own,zero-response,and all3 other-baseline or other-input responses. Cross only one factor at a time. Signs +/-0.1.','references':'KL(own analytic-response intervention || candidate) across full vocabulary, zero-response control. Own-vs-natural replay measured separately. Zero response retains original upstream intervention and other paths.','selection':'No refits or selected reference; all three alternative natural references included.','limitation':'Local linear response replacements and hybrid interventions; not baseline activation swaps or whole-model nonlinear operator recovery.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 outputs={};zeroerrors=[];forwards=0
 for gi,g in enumerate(groups):
  source=int(g['site'][-1]);layer=source+1;rec=g['recipients'];pos=g['position'];native=trace['captures'][gi,'native'][layer,'mlp'];jobs=[]
  for mode in ['block','random']:
   for a in range(4):
    for amp in [-.1,.1]:
     jobs.extend([(mode,a,'own',a,amp),(mode,a,'zero',a,amp)])
     for b in range(4):
      if a!=b:jobs.extend([(mode,a,'baseline',b,amp),(mode,a,'input',b,amp)])
  jobs.extend([('block',a,'own',a,0.) for a in range(4)])
  for st in range(0,len(jobs),8):
   bb=jobs[st:st+8];patch=[];response=[]
   for mode,a,arm,b,amp in bb:
    patch.append(g['directions'][mode]*amp);G=grid['responses'][gi,mode];v=G[a,a] if arm=='own' else torch.zeros_like(G[a,a]) if arm=='zero' else G[b,a] if arm=='baseline' else G[a,b];response.append(native[a].double()+amp*v)
   patch=torch.stack(patch);response=torch.stack(response)
   def inject(m,args,out):
    x,bus=out;x=x.clone();x[:,-1]+=patch.to(x);return x,bus
   def mlp(m,args,out):
    y=out.clone();y[:,-1]=response.to(y);return y
   hs=[model.blocks[source].register_forward_hook(inject),model.blocks[layer].mlp.register_forward_hook(mlp)]
   try:z,_=run(model,torch.stack([panel['all_token_ids'][rec[a]//16,:pos+1] for _,a,_,_,_ in bb]).cuda())
   finally:
    for h in hs:h.remove()
   for n,key in enumerate(bb):
    outputs[(gi,*key)]=z[n].cpu()
    if key[-1]==0:zeroerrors.append(float((z[n].cpu()-prior['native'][gi,rec[key[1]]]).abs().max()))
   forwards+=len(bb)
  print('DONE',gi,flush=True)
 records=[];faith=[]
 for gi,g in enumerate(groups):
  for mode in ['block','random']:
   for a,e in enumerate(g['recipients']):
    for amp in [-.1,.1]:
     own=outputs[gi,mode,a,'own',a,amp].double();lp=own.log_softmax(0);nat=prior['outputs'][gi,e,mode,amp].double().log_softmax(0);faith.append({'group':gi,'mode':mode,'recipient':a,'amp':amp,'own_to_natural_KL':float((lp.exp()*(lp-nat)).sum())})
     for arm in ['zero','baseline','input']:
      for b in ([a] if arm=='zero' else [b for b in range(4) if b!=a]):
       cand=outputs[gi,mode,a,arm,b,amp].double().log_softmax(0);records.append({'group':gi,'site':g['site'],'block':g['block'],'mode':mode,'recipient':a,'arm':arm,'reference':b,'amp':amp,'KL_own_to_candidate':float((lp.exp()*(lp-cand)).sum()),'TV':float((lp.exp()-cand.exp()).abs().sum()/2)})
 summary=[];groupstats=[]
 for gi,g in enumerate(groups):
  for mode in ['block','random']:
   rr=[r for r in records if r['group']==gi and r['mode']==mode];means={arm:float(np.mean([r['KL_own_to_candidate'] for r in rr if r['arm']==arm])) for arm in ['zero','baseline','input']};groupstats.append({'group':gi,'site':g['site'],'block':g['block'],'mode':mode,**means,'baseline_over_zero':means['baseline']/max(means['zero'],1e-30),'input_over_zero':means['input']/max(means['zero'],1e-30)})
 for site in ['residual0','residual1']:
  for mode in ['block','random']:
   rr=[r for r in groupstats if r['site']==site and r['mode']==mode];means={arm:float(np.mean([r[arm] for r in rr])) for arm in ['zero','baseline','input']};summary.append({'site':site,'mode':mode,'groups':len(rr),**means,'baseline_over_zero':means['baseline']/means['zero'],'input_over_zero':means['input']/means['zero'],'baseline_worse_groups':sum(r['baseline']>r['input'] for r in rr)})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'zero_amplitude_replay_max_abs':max(zeroerrors),'whole_model_forwards':forwards,'test_evaluations':0,'finite':all(np.isfinite(r['KL_own_to_candidate']) for r in records),'minimum_KL':min(r['KL_own_to_candidate'] for r in records)};assert audits['hashes_unchanged'] and max(zeroerrors)<1e-4 and audits['finite'] and audits['minimum_KL']>=-1e-12;torch.save({'outputs':outputs,'groups':groups},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'faithfulness':faith,'groupstats':groupstats,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits},indent=2),flush=True)
if __name__=='__main__':main()

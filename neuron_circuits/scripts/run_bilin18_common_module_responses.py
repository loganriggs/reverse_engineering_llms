"""H829 factorial first-block tangent response replacement, retain recipient baselines."""
import json,time,itertools
from pathlib import Path
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
from run_bilin18_early_local_geometry import sha
S='bilin18_common_module_responses'
def cos(a,b):return float(a@b/(a.norm()*b.norm()).clamp_min(1e-30))
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'trace':ART/'bilin18_fixed_direction_depth_captures.pt','prior':ART/'bilin18_fixed_direction_recipients_captures.pt','panel':ART/'bilin18_fresh_early_geometry_panel_captures.pt','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};trace=torch.load(paths['trace'],weights_only=False);prior=torch.load(paths['prior'],weights_only=False);panel=torch.load(paths['panel'],weights_only=False);groups=trace['groups'];arms=['own','common_attention','common_mlp','common_both'];cfg={'experiment':'H829','hashes':hashes,'groups':'All13 H827/H828 groups unchanged, block and random directions.','design':'At first downstream block, replace BOTH attention and MLP outputs at final position by recipient native output + signed amplitude times stored H828 central derivative. Factorial: each derivative own versus group mean. Mean is same vector across4 recipients; baseline and skip paths remain recipient-specific.','dose':[-.1,.1],'control':'Own/own linearized replay vs natural H827 central responses; amp0 own replacements vs native logits.','interpretation':'Controlled tangent response contribution experiment, not natural mediation: MLP replacement does not adapt to modified attention. Common-both block response alignment is imposed algebraically, downstream propagation remains a test.','metric':'Cross-recipient cosines for target residual and final uniformly centered/common-reference-weighted logits; common p is group mean native p.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 outputs={};target={};zero=[];forwards=0;records=[];faith=[];native_replacement=[]
 for gi,g in enumerate(groups):
  source=int(g['site'][-1]);layer=source+1;rec=g['recipients'];pos=g['position'];base=trace['captures'][gi,'native'];jobs=[('block','own',0.)]+[(mode,arm,amp) for mode in ['block','random'] for arm in arms for amp in [-.1,.1]]
  for mode,arm,amp in jobs:
   deriv=trace['captures'][gi,mode];da=deriv[layer,'attention'];dm=deriv[layer,'mlp'];da=da.mean(0,keepdim=True).expand_as(da) if arm in ['common_attention','common_both'] else da;dm=dm.mean(0,keepdim=True).expand_as(dm) if arm in ['common_mlp','common_both'] else dm;vec=g['directions'][mode]*amp;capt={}
   def inject(m,args,out):
    x,bus=out;x=x.clone();x[:,-1]+=vec.to(x);return x,bus
   def attn(m,args,out):
    y,bus=out;y=y.clone();y[:,-1]=(base[layer,'attention'].double()+amp*da).to(y);return y,bus
   def mlp(m,args,out):
    y=out.clone();y[:,-1]=(base[layer,'mlp'].double()+amp*dm).to(y);return y
   def read(m,args,out):capt['target']=out[0][:,-1].cpu()
   hs=[model.blocks[source].register_forward_hook(inject),model.blocks[layer].attn.register_forward_hook(attn),model.blocks[layer].mlp.register_forward_hook(mlp),model.blocks[layer].register_forward_hook(read)]
   try:z,_=run(model,torch.stack([panel['all_token_ids'][e//16,:pos+1] for e in rec]).cuda())
   finally:
    for h in hs:h.remove()
   outputs[gi,mode,arm,amp]=z.cpu();target[gi,mode,arm,amp]=capt['target'];forwards+=len(rec)
   if amp==0:zero.extend(float((z[n].cpu()-prior['native'][gi,e]).abs().max()) for n,e in enumerate(rec))
  p=torch.stack([prior['native'][gi,e].double().softmax(0) for e in rec]).mean(0)
  for mode in ['block','random']:
   natural=torch.stack([(prior['outputs'][gi,e,mode,.1].double()-prior['outputs'][gi,e,mode,-.1].double())/.2 for e in rec])
   for arm in arms:
    final=(outputs[gi,mode,arm,.1].double()-outputs[gi,mode,arm,-.1].double())/.2;internal=(target[gi,mode,arm,.1].double()-target[gi,mode,arm,-.1].double())/.2
    metrics={'target_residual':internal,'uniform_logits':final-final.mean(1,keepdim=True),'common_fisher_logits':(final-(final*p).sum(1,keepdim=True))*p.sqrt()}
    for metric,v in metrics.items():
     cc=[cos(v[a],v[b]) for a,b in itertools.combinations(range(4),2)];records.append({'group':gi,'site':g['site'],'block':g['block'],'mode':mode,'arm':arm,'metric':metric,'pair_cosines':cc,'median_cosine':float(np.median(cc)),'norms':v.norm(dim=1).tolist()})
     if arm=='own':
      ref=trace['captures'][gi,mode][layer,'residual'] if metric=='target_residual' else natural-natural.mean(1,keepdim=True) if metric=='uniform_logits' else (natural-(natural*p).sum(1,keepdim=True))*p.sqrt();faith.append({'group':gi,'site':g['site'],'mode':mode,'metric':metric,'relative_errors':((v-ref).norm(dim=1)/ref.norm(dim=1).clamp_min(1e-30)).tolist(),'cosines':[cos(a,b) for a,b in zip(v,ref)]})
  print('DONE',gi,flush=True)
 summary=[]
 for site in ['residual0','residual1']:
  for mode in ['block','random']:
   for arm in arms:
    summary.append({'site':site,'mode':mode,'arm':arm,**{metric:float(np.median([x for r in records if r['site']==site and r['mode']==mode and r['arm']==arm and r['metric']==metric for x in r['pair_cosines']])) for metric in ['target_residual','uniform_logits','common_fisher_logits']}})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'zero_replacement_max_logit_error':max(zero),'whole_model_forwards':forwards,'test_evaluations':0,'finite':all(np.isfinite(r['median_cosine']) for r in records)};assert audits['hashes_unchanged'] and max(zero)<1e-4 and audits['finite'];torch.save({'outputs':outputs,'target_residual':target,'groups':groups},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'faithfulness':faith,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits},indent=2),flush=True)
if __name__=='__main__':main()

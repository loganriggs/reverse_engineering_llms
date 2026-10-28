"""H828 trace frozen context crossing through downstream stages."""
import json,time,itertools
from pathlib import Path
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
from run_bilin18_early_local_geometry import sha
S='bilin18_fixed_direction_depth'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'prior':ART/'bilin18_fixed_direction_recipients_captures.pt','panel':ART/'bilin18_fresh_early_geometry_panel_captures.pt','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};d=torch.load(paths['prior'],weights_only=False);panel=torch.load(paths['panel'],weights_only=False);groups=d['groups'];cfg={'experiment':'H828','hashes':hashes,'design':'All13 H827 groups, identical recipients and block/random directions, central differences +/-0.1. Native trace included. Final-position traces after every downstream block plus attention RMS input/output and MLP raw/RMS input/output. Final norm, unembedding and softcap traced.','interpretation':'Propagation localization, not module mediation or isolated Jacobian. Module responses see different context-dependent inputs; cosine drops do not quantify causal shares.','metric':'Euclidean response cosines for internal vectors; uniform-centered and common-reference Fisher metric for logits. Every group uses its H827 mean native probability distribution.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 captures={};replay=[];additive=[];identical=[];forwards=0
 for gi,g in enumerate(groups):
  source=int(g['site'][-1]);rec=g['recipients'];pos=g['position'];alltr={}
  for mode,amp in [('zero',0.),('block',-.1),('block',.1),('random',-.1),('random',.1)]:
   tr={};raw={};hs=[];vec=torch.zeros(1152) if mode=='zero' else g['directions'][mode]*amp
   def inject(m,args,out):
    x,bus=out;x=x.clone();x[:,-1]+=vec.to(x);tr[source,'residual']=x[:,-1].cpu();return x,bus
   hs.append(model.blocks[source].register_forward_hook(inject))
   for l in range(source+1,18):
    def bp(m,args,l=l):raw[l]=(m.lambdas[0]*args[0]+m.lambdas[1]*args[2])[:,-1];tr[l,'raw']=raw[l].cpu()
    def ai(m,args,l=l):tr[l,'attention_normalized']=args[0][:,-1].cpu()
    def ao(m,args,out,l=l):tr[l,'attention']=out[0][:,-1].cpu();tr[l,'mlp_raw']=(raw[l]+out[0][:,-1]).cpu()
    def mi(m,args,l=l):tr[l,'mlp_normalized']=args[0][:,-1].cpu()
    def mo(m,args,out,l=l):tr[l,'mlp']=out[:,-1].cpu()
    def bo(m,args,out,l=l):tr[l,'residual']=out[0][:,-1].cpu();additive.append(float((tr[l,'residual']-tr[l,'mlp_raw']-tr[l,'mlp']).abs().max()))
    hs.extend([model.blocks[l].register_forward_pre_hook(bp),model.blocks[l].attn.register_forward_pre_hook(ai),model.blocks[l].attn.register_forward_hook(ao),model.blocks[l].mlp.register_forward_pre_hook(mi),model.blocks[l].mlp.register_forward_hook(mo),model.blocks[l].register_forward_hook(bo)])
   def hi(m,args):tr[18,'final_normalized']=args[0].cpu()
   def ho(m,args,out):tr[18,'pre_softcap']=out.cpu()
   hs.extend([model.lm_head.register_forward_pre_hook(hi),model.lm_head.register_forward_hook(ho)])
   try:z,_=run(model,torch.stack([panel['all_token_ids'][e//16,:pos+1] for e in rec]).cuda())
   finally:
    for h in hs:h.remove()
   tr[18,'logits']=z.cpu();alltr[mode,amp]=tr
   for n,e in enumerate(rec):replay.append(float((z[n].cpu()-d['outputs'][gi,e,mode,amp]).abs().max()))
   forwards+=len(rec)
  for mode in ['block','random']:
   deriv={key:(alltr[mode,.1][key].double()-alltr[mode,-.1][key].double())/.2 for key in alltr[mode,.1]};truth=g['directions'][mode].double();identical.append(float((deriv[source,'residual']-truth).norm(dim=1).max()/truth.norm()));captures[gi,mode]=deriv
  captures[gi,'native']=alltr['zero',0.];print('DONE',gi,flush=True)
 records=[]
 for gi,g in enumerate(groups):
  p=torch.stack([d['native'][gi,e].double().softmax(0) for e in g['recipients']]).mean(0)
  for mode in ['block','random']:
   dd=captures[gi,mode]
   for (layer,stage),values in dd.items():
    choices={'euclidean':values}
    if stage in ['pre_softcap','logits']:choices={'uniform':values-values.mean(1,keepdim=True),'common_fisher':(values-(values*p).sum(1,keepdim=True))*p.sqrt()}
    for metric,v in choices.items():
     cos=[]
     for a,b in itertools.combinations(range(4),2):cos.append(float(v[a]@v[b]/(v[a].norm()*v[b].norm()).clamp_min(1e-30)))
     records.append({'group':gi,'site':g['site'],'block':g['block'],'mode':mode,'layer':layer,'stage':stage,'metric':metric,'pair_cosines':cos,'median_cosine':float(np.median(cos)),'norms':v.norm(dim=1).tolist()})
 summary=[]
 for site,mode,layer,stage,metric in sorted({(r['site'],r['mode'],r['layer'],r['stage'],r['metric']) for r in records}):
  rr=[r for r in records if (r['site'],r['mode'],r['layer'],r['stage'],r['metric'])==(site,mode,layer,stage,metric)];summary.append({'site':site,'mode':mode,'layer':layer,'stage':stage,'metric':metric,'groups':len(rr),'median_cosine':float(np.median([v for r in rr for v in r['pair_cosines']]))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'full_logit_replay_max_abs':max(replay),'residual_additive_identity_max_abs':max(additive),'source_derivative_relative_error_max':max(identical),'whole_model_forwards':forwards,'test_evaluations':0,'finite':all(np.isfinite(r['median_cosine']) for r in records)};assert audits['hashes_unchanged'] and max(replay)<1e-4 and max(identical)<1e-3 and audits['finite'];torch.save({'captures':captures,'groups':groups},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

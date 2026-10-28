"""H786 matched true, within-family shuffled, and isotropic tangent directions."""
import json,math,time,hashlib
from collections import defaultdict
from pathlib import Path
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_bilinear_direction_controls'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'captures':ART/'bilin18_size_controlled_bilinear_transfer_captures.pt','reference':ART/'bilin18_size_controlled_bilinear_transfer.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};c=torch.load(paths['captures'],weights_only=False);old=json.loads(paths['reference'].read_text());examples=c['examples'];inputs=c['inputs'];native=c['native'];cfg={'experiment':'H786','hashes':hashes,'radii':[.05,.15,.3],'causal_radius':.3,'layers':[1,4],'controls':['true','shuffle_next','random_7861','random_7862','random_7863'],'prediction':'If L1 excess error reflects generic layer geometry, it persists in random controls; if donor directions have extra aligned structure they should depart from shuffled or random controls in module geometry and/or downstream answer-margin responses.','design':'Reuse exact H785 base states and all eight families. Shuffle uses next sorted item within same family, same receiver/donor variant, then reprojects displacement onto current base tangent. Isotropic Gaussian directions use three deterministic seeds, normalized after tangent projection. Full and first-order causal interventions at rho .3; geometry at all three radii.','limitations':'Exploratory reused panel. Within-family shuffle retains shared task geometry; random controls do not match activation covariance or output norm. Equal input norm does not imply equal downstream sensitivity. No semantic manifold established.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 families=defaultdict(list)
 for rid in sorted({k[0] for k in examples}):families[examples[rid,0]['family']].append(rid)
 nextid={rid:rr[(i+1)%len(rr)] for rr in families.values() for i,rid in enumerate(rr)};patches={};jobs=[];geometry=[];normerr=[];radiuserr=[];patchreplay=[];directions=[]
 for l in [1,4]:
  m=model.blocks[l].mlp;WL=m.Left.weight.cpu().double();WR=m.Right.weight.cpu().double();WD=m.Down.weight.cpu().double()
  for rid in sorted(nextid):
   for rv,dv,mode in [(0,1,'task'),(1,0,'task'),(0,2,'nuisance')]:
    k=rid,rv;u=inputs[k][l].double();un=u.norm();L=WL@u;R=WR@u;sh=nextid[rid]
    for control in cfg['controls']:
     if control=='true':t=inputs[rid,dv][l].double()-u
     elif control=='shuffle_next':t=inputs[sh,dv][l].double()-inputs[sh,rv][l].double()
     else:
      seed=int(hashlib.sha256(f'{rid}:{rv}:{dv}:{control}'.encode()).hexdigest()[:12],16);g=torch.Generator().manual_seed(seed);t=torch.randn(u.shape,generator=g,dtype=torch.double)
     t=t-u*(t@u)/un.square();assert t.norm()>1e-10;t=t/t.norm();directions.append({'id':rid,'receiver':rv,'donor':dv,'layer':l,'control':control,'shuffle_id':sh if control=='shuffle_next' else None})
     for rho in cfg['radii']:
      theta=2*math.asin(rho/2);v=math.cos(theta)*u+math.sin(theta)*un*t;du=v-u;normerr.append(float(abs(v.norm()/un-1)));radiuserr.append(float(abs(du.norm()/un-rho)));dL=WL@du;dR=WR@du;linear=WD@(dL*R+L*dR);interaction=WD@(dL*dR);full=linear+interaction
      geometry.append({'id':rid,'receiver':rv,'donor':dv,'family':examples[k]['family'],'mode':mode,'layer':l,'rho':rho,'control':control,'full_squared_norm':float(full.square().sum()),'linear_squared_norm':float(linear.square().sum()),'interaction_squared_norm':float(interaction.square().sum())})
      if control=='true':patchreplay.append(float((full.float()-c['patches'][rid,rv,dv,l,rho,'full']).abs().max()))
      if rho==cfg['causal_radius']:
       for term,p in [('linear',linear),('full',full)]:
        key=(rid,rv,dv,l,control,term);patches[key]=p.float();jobs.append({'key':key,'family':examples[k]['family'],'mode':mode})
 print('geometry ready',len(geometry),'jobs',len(jobs),flush=True)
 for k in sorted(examples):
  if k[1] not in [0,1]:continue
  for l in [1,4]:
   key=(k[0],k[1],k[1],l,'identity','full');patches[key]=torch.zeros(1152);jobs.append({'key':key,'family':examples[k]['family'],'mode':'identity'})
 buckets=defaultdict(list)
 for j in jobs:buckets[j['key'][3],len(examples[j['key'][:2]]['ids'])].append(j)
 records=[];identity=[];forwards=0;oldidx={(r['id'],r['receiver'],r['donor'],r['layer'],r['term']):r['margin_change'] for r in old['records'] if r['rho']==.3};marginreplay=[]
 for (l,length),jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12]
   def patch(m,args,out):
    y=out.clone()
    for i,j in enumerate(bb):y[i,-1]+=patches[j['key']].to(y)
    return y
   h=model.blocks[l].mlp.register_forward_hook(patch)
   try:out,_=run(model,torch.tensor([examples[j['key'][:2]]['ids'] for j in bb],device='cuda'))
   finally:h.remove()
   forwards+=len(bb)
   for i,j in enumerate(bb):
    rid,rv,dv,l,control,term=j['key'];k=rid,rv;p=out[i].cpu();delta=measure(p,examples[k]['groups'])['margin']-measure(native[k],examples[k]['groups'])['margin'];records.append({'id':rid,'receiver':rv,'donor':dv,'family':j['family'],'mode':j['mode'],'layer':l,'control':control,'term':term,'margin_change':delta})
    if control=='identity':identity.append(float((p-native[k]).abs().max()))
    if control=='true':marginreplay.append(abs(delta-oldidx[rid,rv,dv,l,term]))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'constant_norm_max_relative_error':max(normerr),'chord_radius_max_abs_error':max(radiuserr),'H785_patch_max_abs':max(patchreplay),'H785_margin_max_abs':max(marginreplay),'identity_logits_max_abs':max(identity),'finite':all(math.isfinite(r['margin_change']) for r in records),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(normerr+radiuserr)<1e-12 and max(patchreplay)<1e-4 and max(marginreplay+identity)<1e-4
 (ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'competence':old['competence'],'geometry':geometry,'records':records,'direction_manifest':directions,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps(audits),flush=True)
if __name__=='__main__':main()

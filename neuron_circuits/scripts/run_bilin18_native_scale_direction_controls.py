"""H788 native-angle paired directional tests with native input/output references."""
import json,math,time,hashlib
from collections import defaultdict
from pathlib import Path
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_native_scale_direction_controls'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'captures':ART/'bilin18_size_controlled_bilinear_transfer_captures.pt','reference':ART/'bilin18_size_controlled_bilinear_transfer.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};c=torch.load(paths['captures'],weights_only=False);old=json.loads(paths['reference'].read_text());examples=c['examples'];inputs=c['inputs'];native=c['native'];cfg={'experiment':'H788','hashes':hashes,'fractions':[.25,.5,1.0],'layers':[1,4],'controls':['true','shuffle_next','random_7861','random_7862','random_7863'],'prediction':'At fractions of native donor angle, true-direction odd response may recover donor-aligned behavior and exceed random controls; true-versus-shuffled distinguishes item specificity from shared-family structure. Evaluate all families without outcome-based selection.','design':'Exact H785 base states, true/shuffle_next/three seeded random tangent controls. Rotate by .25/.5/1 times each receiver-to-donor native angle, with both signs. Compare native donor MLP input substitution against native output substitution; at true positive fraction1 audit residual norm difference separately. Equal radius across controls for a base, not across layers.','limitations':'Exploratory reused panel. Within-family shuffle retains shared task geometry; random controls do not match activation covariance or output norm. Equal input norm does not imply equal downstream sensitivity. No semantic manifold established.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 families=defaultdict(list)
 for rid in sorted({k[0] for k in examples}):families[examples[rid,0]['family']].append(rid)
 nextid={rid:rr[(i+1)%len(rr)] for rr in families.values() for i,rid in enumerate(rr)};patches={};jobs=[];geometry=[];normerr=[];radiuserr=[];patchreplay=[];directions=[];endpoint_errors=[];norm_ratios=[]
 for l in [1,4]:
  m=model.blocks[l].mlp;WL=m.Left.weight.cpu().double();WR=m.Right.weight.cpu().double();WD=m.Down.weight.cpu().double()
  for rid in sorted(nextid):
   for rv,dv,mode in [(0,1,'task'),(1,0,'task'),(0,2,'nuisance')]:
    k=rid,rv;u=inputs[k][l].double();un=u.norm();L=WL@u;R=WR@u;sh=nextid[rid];d=inputs[rid,dv][l].double();dn=d.norm();angle=math.acos(float((u@d/(un*dn)).clamp(-1,1)));norm_ratios.append(float(abs(dn/un-1)))
    for control in cfg['controls']:
     if control=='true':t=inputs[rid,dv][l].double()-u
     elif control=='shuffle_next':t=inputs[sh,dv][l].double()-inputs[sh,rv][l].double()
     else:
      seed=int(hashlib.sha256(f'{rid}:{rv}:{dv}:{control}'.encode()).hexdigest()[:12],16);g=torch.Generator().manual_seed(seed);t=torch.randn(u.shape,generator=g,dtype=torch.double)
     t=t-u*(t@u)/un.square();assert t.norm()>1e-10;t=t/t.norm();directions.append({'id':rid,'receiver':rv,'donor':dv,'layer':l,'control':control,'shuffle_id':sh if control=='shuffle_next' else None})
     for fraction in cfg['fractions']:
      theta=fraction*angle;rho=2*math.sin(theta/2)
      for sign in [1,-1]:
       v=math.cos(theta)*u+sign*math.sin(theta)*un*t;du=v-u;normerr.append(float(abs(v.norm()/un-1)));radiuserr.append(float(abs(du.norm()/un-rho)));dL=WL@du;dR=WR@du;linear=WD@(dL*R+L*dR);interaction=WD@(dL*dR);full=linear+interaction
       geometry.append({'id':rid,'receiver':rv,'donor':dv,'family':examples[k]['family'],'mode':mode,'layer':l,'fraction':fraction,'sign':sign,'rho':rho,'native_angle':angle,'control':control,'full_squared_norm':float(full.square().sum()),'interaction_squared_norm':float(interaction.square().sum())})
       if control=='true' and fraction==1 and sign==1:endpoint_errors.append(float((v-d*un/dn).norm()/un))
       key=(rid,rv,dv,l,control,fraction,sign);patches[key]=full.float();jobs.append({'key':key,'family':examples[k]['family'],'mode':mode})
    for control in ['native_output','native_input']:
     key=(rid,rv,dv,l,control,1.,1);patches[key]=torch.zeros(1152);jobs.append({'key':key,'family':examples[k]['family'],'mode':mode})
 print('geometry ready',len(geometry),'jobs',len(jobs),flush=True)
 for k in sorted(examples):
  if k[1] not in [0,1]:continue
  for l in [1,4]:
   key=(k[0],k[1],k[1],l,'identity',0.,1);patches[key]=torch.zeros(1152);jobs.append({'key':key,'family':examples[k]['family'],'mode':'identity'})
 buckets=defaultdict(list)
 for j in jobs:buckets[j['key'][3],len(examples[j['key'][:2]]['ids'])].append(j)
 records=[];identity=[];forwards=0
 for (l,length),jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12]
   def patch(m,args,out):
    y=out.clone()
    for i,j in enumerate(bb):
     rid,rv,dv,layer,control,fraction,sign=j['key']
     if control=='native_output':y[i,-1]=c['outputs'][rid,dv][l].to(y)
     elif control!='native_input':y[i,-1]+=patches[j['key']].to(y)
    return y
   def pre(m,args):
    x=args[0].clone()
    for i,j in enumerate(bb):
     rid,rv,dv,layer,control,fraction,sign=j['key']
     if control=='native_input':x[i,-1]=inputs[rid,dv][l].to(x)
    return(x,)
   h=model.blocks[l].mlp.register_forward_hook(patch);hp=model.blocks[l].mlp.register_forward_pre_hook(pre)
   try:out,_=run(model,torch.tensor([examples[j['key'][:2]]['ids'] for j in bb],device='cuda'))
   finally:h.remove();hp.remove()
   forwards+=len(bb)
   for i,j in enumerate(bb):
    rid,rv,dv,l,control,fraction,sign=j['key'];k=rid,rv;p=out[i].cpu();delta=measure(p,examples[k]['groups'])['margin']-measure(native[k],examples[k]['groups'])['margin'];records.append({'id':rid,'receiver':rv,'donor':dv,'family':j['family'],'mode':j['mode'],'layer':l,'control':control,'fraction':fraction,'sign':sign,'margin_change':delta})
    if control=='identity':identity.append(float((p-native[k]).abs().max()))

 idx={(r['id'],r['receiver'],r['donor'],r['layer'],r['control'],r['fraction'],r['sign']):r['margin_change'] for r in records};native_replay=[];endpoint_margins=[]
 for r in records:
  if r['control']=='native_output':
   k=r['id'],r['receiver'],r['donor'],r['layer'];native_replay.append(abs(r['margin_change']-idx[*k,'native_input',1.,1]));endpoint_margins.append(abs(r['margin_change']-idx[*k,'true',1.,1]))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'constant_norm_max_relative_error':max(normerr),'chord_radius_max_abs_error':max(radiuserr),'true_endpoint_vs_norm_scaled_donor_max_relative_L2':max(endpoint_errors),'native_donor_receiver_norm_ratio_max_deviation':max(norm_ratios),'native_input_output_margin_max_abs':max(native_replay),'true_endpoint_native_output_margin_max_abs':max(endpoint_margins),'identity_logits_max_abs':max(identity),'finite':all(math.isfinite(r['margin_change']) for r in records),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(normerr+radiuserr+endpoint_errors)<1e-10 and max(native_replay+identity)<1e-4
 (ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'competence':old['competence'],'geometry':geometry,'records':records,'direction_manifest':directions,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps(audits),flush=True)
if __name__=='__main__':main()

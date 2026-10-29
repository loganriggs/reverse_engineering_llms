"""M67 fixed half-change and doubled-half-change downstream interventions."""
from pathlib import Path
import json,hashlib,torch
from inspect_model import load_model,trace
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def margin(z,g,v):return float(torch.logsumexp(z.double()[g[v]],0)-torch.logsumexp(z.double()[g[1-v]],0))
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);pp=V/'native_change_subset_response_protocol.json';q=json.loads(pp.read_text());ap=V/'native_change_coordination.json'
 for name,key in [('native_change_coordination.json','atlas_sha256'),('native_change_coordination_audit.json','audit_sha256'),('training_causal_coverage.json','training_native_sha256'),('source_holdout_native.json','holdout_native_sha256')]:assert sha(V/name)==q[key]
 atlas=json.loads(ap.read_text());mp=Path(atlas['masks_path']);assert sha(mp)==atlas['masks_sha256'];bank=torch.load(mp,weights_only=True).cuda().float();masks=torch.stack([bank[0],1-bank[0],bank[1],1-bank[1]]);banks={};pairs={}
 for split,pn,rn in [('train','training_causal_coverage_protocol.json','training_causal_coverage.json'),('source_eval','source_holdout_native_protocol.json','source_holdout_native.json')]:
  pr=json.loads((V/pn).read_text());pairs.update({x['id']:x for x in pr['pairs']});rr=json.loads((V/rn).read_text());sp=Path(rr['samples_path']);assert sha(sp)==rr['samples_sha256'];banks[split]=torch.load(sp,weights_only=True)
 op=V/'native_change_subset_responses.json';assert not op.exists();dest=A/'emlp_m67';dest.mkdir(exist_ok=True);progress=A/'emlp_m67_progress.pt';rows=[];teachers=[];files=[];calls=0;completed=0;replays=0
 if progress.exists():
  s=torch.load(progress,weights_only=True);assert s['source_sha256']==sha(Path(__file__)) and s['protocol_sha256']==sha(pp);rows,teachers,files,calls,completed,replays=[s[k] for k in ['rows','teachers','files','calls','completed','replays']]
 model,receipt=load_model();assert receipt.weights_sha256==atlas['checkpoint']['weights_sha256']
 for pi,f in enumerate(atlas['files']):
  if pi<completed:continue
  fp=Path(f['path']);assert sha(fp)==f['sha256'];b=torch.load(fp,weights_only=True);pair=pairs[f['pair_id']];saved=banks[f['split']];blob=dict(pair_id=f['pair_id'],split=f['split'],teachers={},changed={},deltas={},full_deltas={});bases=[]
  for ep in [0,1]:
   ids=torch.tensor([pair['token_ids'][ep]],device='cuda');out,_=trace(model,ids);calls+=1;assert torch.equal(out[0,-1].cpu(),b['logits'][ep]);assert torch.equal(out[0,-1].cpu(),saved['baselines'][pair['id']+f'|{ep}']['logits']);bases.append(out[0,-1])
  for spec in [x for x in atlas['rows'] if x['pair_id']==pair['id']]:
   ident={k:spec[k] for k in ['pair_id','split','family','endpoint','layer','region']}
   if spec['status']=='ineligible':rows.append(ident|dict(status='ineligible',reason=spec['reason']));continue
   ep=spec['endpoint'];l=spec['layer'];pos=spec['recipient_positions'];dp=spec['donor_positions'];ids=torch.tensor([pair['token_ids'][ep]],device='cuda');base=bases[ep];blp=base.double().log_softmax(-1);g=pair['answer_groups'][ep];variant=pair['variants'][ep];bm=margin(base,g,variant);cr=b['captures'][ep][l];cd=b['captures'][1-ep][l];native=(cd['write'][:,dp]-cr['write'][:,pos]).cuda();dc=(cd['products'][:,dp]-cr['products'][:,pos]).cuda();mlp=model.transformer.h[l].mlp;localfull=mlp.Down(dc);sid=f'{ep}|{l}|'+spec['region'];sk=pair['id']+'|'+sid
   def run(delta):
    nonlocal calls
    if not bool(delta.count_nonzero()):return base
    patch=torch.zeros_like(cr['write'],device='cuda');patch[:,pos]=delta;out,_=trace(model,ids,replace=lambda j,z,w:w+patch if j==l else w);calls+=1;return out[0,-1]
   teacher=run(native);tlp=teacher.double().log_softmax(-1);response=tlp-blp;te=float((blp.exp()*response.square()).sum());tm=margin(teacher,g,variant)-bm;ne=float(native.double().square().sum());blob['teachers'][sid]=teacher.cpu();blob['full_deltas'][sid]=native.cpu()
   if sk in saved['changed']:assert torch.equal(teacher.cpu(),saved['changed'][sk]);assert torch.equal(native.cpu(),saved['deltas'][sk]['delta']);replays+=1
   teachers.append(ident|dict(target_energy=te,margin_change=tm,delta_energy=ne,native_margin=bm,native_binary=bm>0,native_top1=int(base.argmax()) in g[variant],full_product_difference_error=float((localfull.double()-native.double()).square().sum())))
   for mi,mask in enumerate(masks):
    half=mlp.Down(dc*mask)
    for factor in [1,2]:
     delta=half*factor;z=run(delta);response_error=float((blp.exp()*(z.double().log_softmax(-1)-blp-response).square()).sum());me=(margin(z,g,variant)-bm-tm)**2;key=sid+f'|{mi}|{factor}';blob['changed'][key]=z.cpu();blob['deltas'][key]=delta.cpu();rows.append(ident|dict(status='evaluated',mask=mi,factor=factor,response_error=response_error,target_energy=te,margin_error=me,margin_target_energy=tm**2,local_error=float((delta.double()-native.double()).square().sum()),native_delta_energy=ne,delta_energy=float(delta.double().square().sum())))
  path=dest/fp.name;torch.save(blob,path);files.append(dict(pair_id=pair['id'],split=f['split'],path=str(path),sha256=sha(path)));tmp=progress.with_suffix('.tmp');torch.save(dict(source_sha256=sha(Path(__file__)),protocol_sha256=sha(pp),rows=rows,teachers=teachers,files=files,calls=calls,completed=pi+1,replays=replays),tmp);tmp.replace(progress);print('PAIRS',pi+1,'FORWARDS',calls,flush=True)
 assert len(rows)==12054 and len(teachers)==1506 and replays==1338
 op.write_text(json.dumps(dict(status='completed',source_sha256=sha(Path(__file__)),protocol_sha256=sha(pp),atlas_sha256=sha(ap),rows=rows,teachers=teachers,files=files,forward_count=calls,old_teacher_replays=replays),indent=2)+'\n');print('DONE',calls,flush=True)
if __name__=='__main__':main()

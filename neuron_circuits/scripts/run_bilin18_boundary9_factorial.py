"""H638 full receiver/donor factorial at block9 boundary and final readout."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_boundary9_factorial'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_cross_task_response_panel_manifest.json','capture':ART/'bilin18_matched_mlp5_contexts_captures.pt','source':ART/'bilin18_cross_task_response_panel_captures.pt','internal':ART/'bilin18_internal_response_subspaces_captures.pt','runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['capture'],weights_only=False);source=torch.load(paths['source'],weights_only=False);internal=torch.load(paths['internal'],weights_only=False);names=['original','span2','orthogonal2'];caches=cap['caches'];bus=cap['bus'];x0=cap['x0'];lookup={(e['family'],e['form'],e['item']):i for i,e in enumerate(rows)};donors={kind:[lookup[e['family'],1-e['form'],e['item']] if kind=='cross' else lookup[e['family'],e['form'],(e['item']+4)%8] for e in rows] for kind in ['cross','within']};assert all(all(jj[jj[i]]==i for i in range(len(rows))) for jj in donors.values());cfg=dict(experiment='H638',hashes=hashes,probes=names,axes={'1':'block9 incoming tangent','2':'downstream block10-17 Jacobians','4':'final RMS and softcap readout Jacobian'},design='Full2x2x2 factorial of receiver/donor components for both involutive donor maps. Same native fixed contexts. All-donor endpoint recovery required. Incoming block9 tangents centrally measured at .02. No fitting.',limitations='Hybrid local linear maps need not be jointly realizable. Norm effects do not add; exact vector interaction decomposition reported. No task-accuracy claim.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
 for i,e in enumerate(rows):buckets[len(e['ids'])].append(i)
 jvps=0;finals={}
 for name in names:
  finals[name]={}
  for kind in ['native','cross','within']:
   incoming=internal['tangents'][name][9];d=incoming if kind=='native' else incoming[donors[kind]]
   for l in range(10,18):
    result=torch.empty_like(d)
    for ii in buckets.values():
     for st in range(0,len(ii),16):
      bb=ii[st:st+16];x=torch.stack([caches[l][i] for i in bb]).cuda();b=torch.stack([bus[i] for i in bb]).cuda();e0=torch.stack([x0[i] for i in bb]).cuda();prefix=x[:,:-1];last=x[:,-1];block=model.blocks[l]
      def fn(tail):return block(torch.cat([prefix,tail[:,None]],1),b,e0)[0][:,-1]
      _,out=torch.func.jvp(fn,(last,),(d[bb].cuda().float(),));result[bb]=out.cpu().double();jvps+=1
    d=result
   finals[name][kind]=d
  print('PROPAGATED',name,flush=True)
 X=internal['native_captures'][17].double().cuda();W=model.lm_head.weight.double();records=[];interactions=[];valid=[];maxdecomp=0.
 def readout(d,idx):
  x=X[idx];r=(x.square().mean()+torch.finfo(torch.float32).eps).sqrt();sat=1-torch.tanh((x/r)@W.T/30).square();return (sat*((d.cuda()/r-x*(x*d.cuda()).mean()/r.pow(3))@W.T)).cpu()
 for name in names:
  for kind,jj in donors.items():
   for i,e in enumerate(rows):
    j=jj[i];p=source['native_logits'][i].double().softmax(0)
    def center(v):return v-(p*v).sum()
    def norm(v):return (center(v).square()*p).sum().sqrt()
    outs={}
    for mask in range(8):
     a=bool(mask&1);b=bool(mask&2);c=bool(mask&4)
     if a==b:d=finals[name]['native'][j if a else i]
     else:d=finals[name][kind][j if b else i]
     outs[mask]=readout(d,j if c else i)
    receiver=source['responses'][name][i];donor=source['responses'][name][j];gap=norm(donor-receiver);valid.append(dict(id=e['id'],probe=name,kind=kind,receiver_error=float(norm(outs[0]-receiver)/norm(receiver)),donor_error=float(norm(outs[7]-donor)/norm(donor)),endpoint_error_over_gap=float(norm(outs[7]-donor)/gap)))
    for mask,h in outs.items():records.append(dict(id=e['id'],family=e['family'],form=e['form'],probe=name,kind=kind,mask=mask,remaining_gap=float(norm(donor-h)/gap),change_over_gap=float(norm(h-outs[0])/gap),target_derivative=float(center(h)[e['answer_id']])))
    mob={0:outs[0]}
    for mask in range(1,8):mob[mask]=outs[mask]-sum(mob[sub] for sub in mob if sub&mask==sub)
    maxdecomp=max(maxdecomp,float((sum(mob.values())-outs[7]).abs().max()))
    additive=outs[0]+mob[1]+mob[2]+mob[4]
    interactions.append(dict(id=e['id'],family=e['family'],form=e['form'],probe=name,kind=kind,additive_remaining_gap=float(norm(donor-additive)/gap),**{f'interaction_{mask}_over_gap':float(norm(mob[mask])/gap) for mask in [3,5,6,7]}))
 audits=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},new_whole_model_forwards=0,batched_jvp_calls=jvps,receiver_max_error=max(v['receiver_error'] for v in valid),donor_max_error=max(v['donor_error'] for v in valid),endpoint_max_error_over_gap=max(v['endpoint_error_over_gap'] for v in valid),decomposition_max_abs=maxdecomp,finite=all(np.isfinite(v['remaining_gap']) for v in records));torch.save(dict(final_tangents=finals),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,interactions=interactions,validation=valid,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()

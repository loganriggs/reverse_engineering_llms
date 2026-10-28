"""EMLP-P3: fixed token/context crossing and MLP1 local/future response factorial."""
import argparse,json,hashlib
from pathlib import Path
import torch
from inspect_model import load_model,encode,trace

CONTEXTS=[('The dog waited near the','The dogs waited near the'),
 ('She left a note beside the','He left a note beside the'),
 ('I enjoyed the walk to the','I disliked the walk to the'),
 ('Yesterday we visited the','Today we visited the')]
ANCHORS=[' house',' park',' school',' station']
CONTINUATION='. It was quiet outside.'
ARMS=('frozen_all','local_only','future_only','live')

def position_kl(base,edited):
 a=base.double().log_softmax(-1);b=edited.double().log_softmax(-1)
 return (a.exp()*(a-b)).sum(-1)[0]

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');model,receipt=load_model()
 device=next(model.parameters()).device;rows=[];panels=[];calls=0;identity_error=0.;prefix_error=0.;norm_error=0.;samples=[]
 for ci,contexts in enumerate(CONTEXTS):
  for ai,anchor in enumerate(ANCHORS):
   states=[]
   for context in contexts:
    prefix=context+anchor;text=prefix+CONTINUATION;ids=encode(text,device);pids=encode(prefix,device)
    assert torch.equal(ids[:,:pids.shape[1]],pids)
    pos=pids.shape[1]-1;lg,c=trace(model,ids);calls+=1
    ident,_=trace(model,ids,replace=lambda i,z,y:c[1]['native_write'] if i==1 else y);calls+=1
    identity_error=max(identity_error,float((lg-ident).abs().max()))
    states.append((ids,pos,lg,c))
   assert states[0][0][0,states[0][1]]==states[1][0][0,states[1][1]]
   panels.append(dict(context=ci,anchor=anchor,texts=[s+anchor+CONTINUATION for s in contexts],positions=[s[1] for s in states],token_id=int(states[0][0][0,states[0][1]])))
   for rec in (0,1):
    ids,pos,base,c=states[rec];_,dp,_,dc=states[1-rec];fixed=c[1]['native_write']
    delta=dc[0]['native_write'][:,dp]-c[0]['native_write'][:,pos]
    for control in range(5):
     if control==0:edit=delta;kind='context'
     else:
      g=torch.Generator(device=device).manual_seed(2026102900+ci*1000+ai*100+rec*10+control)
      perm=torch.randperm(delta.shape[-1],device=device,generator=g)
      sign=torch.randint(0,2,delta.shape,device=device,generator=g)*2-1;edit=delta[:,perm]*sign;kind='signed_permutation'
     ne=float(abs(edit.norm()-delta.norm())/delta.norm().clamp_min(1e-12));norm_error=max(norm_error,ne);assert ne<1e-6
     metrics={};outputs={}
     for arm in ARMS:
      def replace(i,z,y):
       if i==0:
        out=y.clone();out[:,pos]+=edit;return out
       if i!=1 or arm=='live':return y
       out=y.clone()
       if arm=='frozen_all':return fixed
       if arm=='local_only':out[:,pos+1:]=fixed[:,pos+1:]
       if arm=='future_only':out[:,pos]=fixed[:,pos]
       return out
      lg,_=trace(model,ids,replace=replace);calls+=1;outputs[arm]=lg[:,pos:].detach().cpu()
      before=float((lg[:,:pos]-base[:,:pos]).abs().max());prefix_error=max(prefix_error,before)
      k=position_kl(base[:,pos:],lg[:,pos:]);metrics[arm]=dict(at_edit_KL=float(k[0]),future_mean_KL=float(k[1:].mean()),last_KL=float(k[-1]),per_position_KL=k.cpu().tolist())
     # Output-space interaction of local and future MLP1 write responses.
     inter=outputs['live']-outputs['local_only']-outputs['future_only']+outputs['frozen_all']
     inter=inter-inter.mean(-1,keepdim=True)
     response=outputs['live']-outputs['frozen_all'];response=response-response.mean(-1,keepdim=True)
     rows.append(dict(context=ci,anchor=anchor,receiver=rec,kind=kind,control=control,position=pos,
       edit_norm=float(edit.norm()),metrics=metrics,future_interaction_rms=float(inter[:,1:].square().mean().sqrt()),
       future_total_response_rms=float(response[:,1:].square().mean().sqrt())))
     if ci==0 and ai==0 and rec==0 and control in (0,1):
      samples.append(dict(row_index=len(rows)-1,base=base[:,pos:].cpu(),outputs=outputs))
   print(f'completed context {ci+1}/4 anchor {anchor}',flush=True)
 assert identity_error==0 and prefix_error<1e-4
 def summarize(rr):
  return dict(n=len(rr),means={a:{region:sum(r['metrics'][a][region] for r in rr)/len(rr) for region in ['at_edit_KL','future_mean_KL','last_KL']} for a in ARMS},
   live_lower_than_frozen={region:sum(r['metrics']['live'][region]<r['metrics']['frozen_all'][region] for r in rr) for region in ['at_edit_KL','future_mean_KL','last_KL']})
 summary={kind:summarize([r for r in rows if r['kind']==kind]) for kind in ['context','signed_permutation']}
 by_anchor={a:{k:summarize([r for r in rows if r['anchor']==a and r['kind']==k]) for k in summary} for a in ANCHORS}
 payload=dict(status='completed_exploratory',checkpoint_sha256=receipt.weights_sha256,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  source_layer=0,response_layer=1,comparison_panel='EMLP-P2 opened prompts, reused seeds; different source-layer delta norms',
  protocol='Fixed 4 context pairs x 4 location-noun anchors. Bidirectional same-token write swaps at MLP0, plus four signed-permutation controls. Shared continuation. MLP1 response factorial: frozen_all/local_only/future_only/live; all unedited baselines identical. Native recipient x0 and value bus retained. No fitting or task labels.',
  panels=panels,arms=ARMS,forward_count=calls,identity_max_error=identity_error,pre_edit_max_logit_error=prefix_error,max_norm_match_error=norm_error,summary=summary,by_anchor=by_anchor,rows=rows,
  limitations=['Fixed small authored panel, all four anchors are location nouns; no natural corpus inference.','Absolute position is not independently randomized. Future effects tested using one shared continuation.','Dependent directions and controls; no independent-sample confidence intervals.','Oracle intervention; output change is not necessarily behavioral damage.'])
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2)+'\n')
 torch.save(samples,Path('/tmp/emlp_p3_audit_logits.pt'))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

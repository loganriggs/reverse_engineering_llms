"""EMLP-P5: cross prefix MLP0 change with its induced all-position MLP1 write response."""
import argparse,json,hashlib
from pathlib import Path
import torch
from inspect_model import load_model,encode,trace
from probe_mlp2_token_transfer import CONTEXTS,ANCHORS,CONTINUATION,position_kl

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');m,receipt=load_model();device=next(m.parameters()).device
 rows=[];calls=0;identity=0.;joint_error=0.;samples=[];panel=[]
 for ci,contexts in enumerate(CONTEXTS):
  for ai,anchor in enumerate(ANCHORS):
   states=[]
   for text in contexts:
    pid=encode(text+anchor,device);ids=encode(text+anchor+CONTINUATION,device);pos=pid.shape[1]-1
    assert torch.equal(ids[:,:pid.shape[1]],pid)
    logits,c=trace(m,ids);calls+=1;states.append((ids,pos,logits,c))
   a,p,_,_=states[0];b,q,_,_=states[1]
   assert a.shape==b.shape and p==q and int((a!=b).sum())==1 and int((a!=b).nonzero()[0,1])<p
   panel.append(dict(context=ci,anchor=anchor,position=p,texts=[x+anchor+CONTINUATION for x in contexts]))
   for rec in (0,1):
    ids,pos,baseline,c=states[rec];_,_,donor,dc=states[1-rec];fixed=c[1]['native_write']
    ident,_=trace(m,ids,replace=lambda i,z,y:fixed if i==1 else y);calls+=1;identity=max(identity,float((ident-baseline).abs().max()))
    raw_delta=dc[0]['native_write'][:,:pos+1]-c[0]['native_write'][:,:pos+1]
    for control in range(5):
     if control==0:delta=raw_delta;kind='context'
     else:
      g=torch.Generator(device=device).manual_seed(2026103000+ci*1000+ai*100+rec*10+control)
      perm=torch.randperm(raw_delta.shape[-1],device=device,generator=g)
      signs=(torch.randint(0,2,(1,1,raw_delta.shape[-1]),device=device,generator=g)*2-1)
      delta=raw_delta[:,:,perm]*signs;kind='random_prefix'
      assert torch.allclose(delta.norm(dim=-1),raw_delta.norm(dim=-1),rtol=1e-6,atol=1e-6)
     def prefix(i,z,y):
      if i!=0:return y
      out=y.clone();out[:,:pos+1]+=delta;return out
     joint,jc=trace(m,ids,replace=prefix);calls+=1;induced=jc[1]['native_write']
     def direct(i,z,y):return fixed if i==1 else prefix(i,z,y)
     direct_logits,_=trace(m,ids,replace=direct);calls+=1
     response_logits,_=trace(m,ids,replace=lambda i,z,y:induced if i==1 else y);calls+=1
     replay,_=trace(m,ids,replace=lambda i,z,y:induced if i==1 else prefix(i,z,y));calls+=1
     joint_error=max(joint_error,float((replay-joint).abs().max()))
     outputs=dict(baseline=baseline,bypass_only=direct_logits,response_only=response_logits,joint=joint)
     metrics={}
     for arm,lg in outputs.items():
      gap=position_kl(donor[:,pos:],lg[:,pos:]);change=position_kl(baseline[:,pos:],lg[:,pos:])
      metrics[arm]=dict(donor_gap_at_edit=float(gap[0]),donor_gap_future=float(gap[1:].mean()),
       recipient_change_at_edit=float(change[0]),recipient_change_future=float(change[1:].mean()))
     interaction=joint-direct_logits-response_logits+baseline
     interaction-=interaction.mean(-1,keepdim=True)
     effect=joint-baseline;effect=effect-effect.mean(-1,keepdim=True)
     rows.append(dict(context=ci,anchor=anchor,receiver=rec,control=control,kind=kind,
      prefix_edit_norm=float(delta.norm()),metrics=metrics,
      interaction_future_rms=float(interaction[:,pos+1:].square().mean().sqrt()),joint_future_rms=float(effect[:,pos+1:].square().mean().sqrt())))
     if ci==0 and ai==0 and rec==0 and control in (0,1):
      samples.append(dict(row_index=len(rows)-1,donor=donor[:,pos:].cpu(),outputs={k:v[:,pos:].cpu() for k,v in outputs.items()}))
   print('finished',ci,anchor,flush=True)
 assert identity==joint_error==0
 def summarize(rr):
  keys=list(rr[0]['metrics']);means={a:{metric:sum(r['metrics'][a][metric] for r in rr)/len(rr) for metric in rr[0]['metrics'][a]} for a in keys}
  return dict(n=len(rr),means=means,remaining_donor_gap={a:{region:means[a]['donor_gap_'+region]/means['baseline']['donor_gap_'+region] for region in ['at_edit','future']} for a in keys})
 summary={k:summarize([r for r in rows if r['kind']==k]) for k in ['context','random_prefix']}
 by_context={str(ci):summarize([r for r in rows if r['kind']=='context' and r['context']==ci]) for ci in range(4)}
 result=dict(status='completed_exploratory',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),panel_source_sha256=hashlib.sha256((Path(__file__).parent/'probe_mlp2_token_transfer.py').read_bytes()).hexdigest(),checkpoint_sha256=receipt.weights_sha256,
  protocol='Native recipient persistent state retained. MLP0 prefix edit crossed with baseline versus induced all-position MLP1 write. Both-prefix-and-response joint replay exact. Four shared signed-permutation maps per recipient preserve each prefix position edit norm. Output target is natural donor distribution.',
  panel=panel,forward_count=calls,identity_error=identity,joint_replay_error=joint_error,summary=summary,by_context=by_context,rows=rows,
  limitations=['Opened small authored panel, not fresh confirmation.','Mediation is conditional on these physical interventions; nonlinear output metrics have no unique additive shares.','Response-only uses native induced MLP1 writes as an oracle and is not a standalone generator.','Random prefix changes preserve support and per-position norms but are not guaranteed on-manifold.'])
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');torch.save(samples,'/tmp/emlp_p5_logits.pt');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

"""EMLP-P4: finite transfer of MLP0 writes versus the natural MLP1 context change."""
import argparse,json,hashlib
from pathlib import Path
import torch
from inspect_model import load_model,encode,trace
from probe_mlp2_token_transfer import CONTEXTS,ANCHORS,CONTINUATION,position_kl


def vector_metrics(target,predicted):
 t=target.double().flatten();p=predicted.double().flatten();t2=float(t@t);p2=float(p@p);dot=float(t@p)
 return dict(target_sqnorm=t2,response_sqnorm=p2,dot=dot,error_sqnorm=float((p-t).square().sum()),
  signed_projection=dot/max(t2,1e-30),cosine=dot/max((t2*p2)**.5,1e-30),norm_ratio=(p2/max(t2,1e-30))**.5,
  relative_L2=(float((p-t).square().sum())/max(t2,1e-30))**.5)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');model,receipt=load_model();device=next(model.parameters()).device
 rows=[];panels=[];vectors=[];calls=0;identity_error=0.;oracle_error=0.
 for ci,contexts in enumerate(CONTEXTS):
  for ai,anchor in enumerate(ANCHORS):
   states=[]
   for context in contexts:
    prefix=context+anchor;ids=encode(prefix+CONTINUATION,device);pid=encode(prefix,device)
    assert torch.equal(ids[:,:pid.shape[1]],pid);pos=pid.shape[1]-1
    lg,c=trace(model,ids);calls+=1
    identity,_=trace(model,ids,replace=lambda i,z,y:c[0]['native_write'] if i==0 else y);calls+=1
    identity_error=max(identity_error,float((lg-identity).abs().max()));states.append((ids,pos,lg,c))
   ids0,p0,_,_=states[0];ids1,p1,_,_=states[1]
   aligned=ids0.shape==ids1.shape and p0==p1
   mismatch=(ids0!=ids1).nonzero()[:,1].tolist() if aligned else []
   aligned=aligned and len(mismatch)==1 and mismatch[0]<p0
   panels.append(dict(context=ci,anchor=anchor,positions=[p0,p1],prefix_transfer_eligible=aligned,changed_token_positions=mismatch))
   for rec in (0,1):
    ids,pos,base,c=states[rec];dids,dp,donor,dc=states[1-rec]
    assert ids[0,pos]==dids[0,dp]
    delta=dc[0]['native_write'][:,dp]-c[0]['native_write'][:,pos]
    target=dc[1]['native_write'][:,dp]-c[1]['native_write'][:,pos]
    natural_kl=position_kl(donor[:,dp:],base[:,pos:])
    arms=[('local_MLP0',0),('donor_MLP1_oracle',0)]+([('prefix_MLP0',0)] if aligned else [])+[('random_local_MLP0',k) for k in range(1,5)]
    for arm,control in arms:
     edit=delta
     if control:
      g=torch.Generator(device=device).manual_seed(2026102900+ci*1000+ai*100+rec*10+control)
      perm=torch.randperm(delta.shape[-1],device=device,generator=g);sign=torch.randint(0,2,delta.shape,device=device,generator=g)*2-1
      edit=delta[:,perm]*sign
      assert float(abs(edit.norm()-delta.norm())/delta.norm().clamp_min(1e-12))<1e-6
     def replace(i,z,y):
      if arm=='donor_MLP1_oracle' and i==1:
       out=y.clone();out[:,pos]=dc[1]['native_write'][:,dp];return out
      if arm=='donor_MLP1_oracle' or i!=0:return y
      out=y.clone()
      if arm=='prefix_MLP0':out[:,:pos+1]=dc[0]['native_write'][:,:pos+1]
      else:out[:,pos]+=edit
      return out
     lg,cc=trace(model,ids,replace=replace);calls+=1
     response=cc[1]['effective_write'][:,pos]-c[1]['native_write'][:,pos]
     v=vector_metrics(target,response)
     if arm=='donor_MLP1_oracle':oracle_error=max(oracle_error,v['relative_L2'])
     remaining=position_kl(donor[:,dp:],lg[:,pos:])
     rows.append(dict(context=ci,anchor=anchor,receiver=rec,arm=arm,control=control,vector=v,
      native_donor_gap_KL_at_edit=float(natural_kl[0]),remaining_donor_gap_KL_at_edit=float(remaining[0]),
      native_donor_gap_KL_future=float(natural_kl[1:].mean()),remaining_donor_gap_KL_future=float(remaining[1:].mean())))
     vectors.append(dict(row=len(rows)-1,target=target.cpu(),response=response.cpu()))
   print(f'context {ci+1}, anchor {anchor} done',flush=True)
 assert identity_error==0 and oracle_error==0
 def aggregate(rr):
  tt=sum(r['vector']['target_sqnorm'] for r in rr);pp=sum(r['vector']['response_sqnorm'] for r in rr);dot=sum(r['vector']['dot'] for r in rr);err=sum(r['vector']['error_sqnorm'] for r in rr)
  return dict(n=len(rr),signed_projection=dot/tt,cosine=dot/(tt*pp)**.5,norm_ratio=(pp/tt)**.5,relative_L2=(err/tt)**.5,write_R2=1-err/tt,
   relative_remaining_donor_KL={x:sum(r['remaining_donor_gap_KL_'+x] for r in rr)/sum(r['native_donor_gap_KL_'+x] for r in rr) for x in ['at_edit','future']})
 arms=sorted(set(r['arm'] for r in rows));summary={a:aggregate([r for r in rows if r['arm']==a]) for a in arms}
 by_context={str(ci):{a:aggregate([r for r in rows if r['arm']==a and r['context']==ci]) for a in arms if any(r['arm']==a and r['context']==ci for r in rows)} for ci in range(4)}
 payload=dict(status='completed_exploratory',checkpoint_sha256=receipt.weights_sha256,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  panel_source_sha256=hashlib.sha256((Path(__file__).parent/'probe_mlp2_token_transfer.py').read_bytes()).hexdigest(),
  protocol='Opened P2/P3 panel. MLP0 local or aligned-prefix write transplant; all recipient persistent state retained. Natural donor MLP1 write difference is the target. Four normmatched local random controls and direct donor MLP1 write oracle. Prefix eligibility requires equal token lengths/anchor indices and exactly one earlier token difference.',
  forward_count=calls,identity_max_error=identity_error,oracle_relative_error=oracle_error,panels=panels,summary=summary,by_context=by_context,rows=rows,
  limitations=['Finite conditional transfer, not unique causal mediation decomposition.','Local and prefix edits have different support and norm; not a budget-matched competition.','Other context routes remain recipient-native; low recovery does not establish independence.','Reused authored panel, no fresh validation or task accuracy claim.','Pooled vector scores are energy weighted; signed projection is not a variance-explained percentage.'])
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2)+'\n')
 torch.save(vectors,Path('/tmp/emlp_p4_vectors.pt'));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

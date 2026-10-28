"""EMLP-P6: changed-word versus other-position components of the induced MLP1 response."""
import argparse,json,hashlib
from pathlib import Path
import torch
from inspect_model import load_model,encode,trace
from probe_mlp2_token_transfer import CONTEXTS,ANCHORS,CONTINUATION,position_kl

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');m,receipt=load_model();device=next(m.parameters()).device
 rows=[];directions=[];calls=0;identity=0.;before_error=0.;samples=[]
 for ci,contexts in enumerate(CONTEXTS):
  for ai,anchor in enumerate(ANCHORS):
   states=[]
   for text in contexts:
    pid=encode(text+anchor,device);ids=encode(text+anchor+CONTINUATION,device);pos=pid.shape[1]-1
    assert torch.equal(ids[:,:pid.shape[1]],pid)
    lg,c=trace(m,ids);calls+=1;states.append((ids,pos,lg,c))
   a,p,_,_=states[0];b,q,_,_=states[1]
   assert a.shape==b.shape and p==q and int((a!=b).sum())==1
   changed=int((a!=b).nonzero()[0,1]);assert changed<p
   for rec in (0,1):
    ids,pos,base,c=states[rec];_,_,donor,dc=states[1-rec];fixed=c[1]['native_write']
    ident,_=trace(m,ids,replace=lambda i,z,y:fixed if i==1 else y);calls+=1;identity=max(identity,float((ident-base).abs().max()))
    delta0=dc[0]['native_write'][:,:pos+1]-c[0]['native_write'][:,:pos+1]
    def prefix(i,z,y):
     if i!=0:return y
     out=y.clone();out[:,:pos+1]+=delta0;return out
    _,induced=trace(m,ids,replace=prefix);calls+=1
    new=induced[1]['native_write'];delta=new-fixed
    if changed:before_error=max(before_error,float(delta[:,:changed].abs().max()))
    energies=delta.double().square().sum(-1)[0]
    directions.append(dict(context=ci,anchor=anchor,receiver=rec,changed_position=changed,anchor_position=pos,
      changed_word_response_energy=float(energies[changed]),total_response_energy=float(energies.sum()),
      per_position_response_energy=energies.cpu().tolist(),recipient_ids=ids.cpu().tolist(),donor_ids=states[1-rec][0].cpu().tolist()))
    baseline_gap=position_kl(donor[:,pos:],base[:,pos:]);outputs={}
    arms=[('changed_word',0),('other_positions',0),('all_positions',0)]+[('random_changed_word',k) for k in range(1,5)]
    for arm,control in arms:
     patched=fixed.clone()
     if arm=='all_positions':patched=new
     elif arm=='changed_word':patched[:,changed]=new[:,changed]
     elif arm=='other_positions':patched=new.clone();patched[:,changed]=fixed[:,changed]
     else:
      e=delta[:,changed];g=torch.Generator(device=device).manual_seed(2026103100+ci*1000+ai*100+rec*10+control)
      perm=torch.randperm(e.shape[-1],device=device,generator=g);sign=torch.randint(0,2,e.shape,device=device,generator=g)*2-1
      random=e[:,perm]*sign;assert torch.allclose(e.norm(),random.norm(),rtol=1e-6,atol=1e-6)
      patched[:,changed]+=random
     lg,_=trace(m,ids,replace=lambda i,z,y:patched if i==1 else y);calls+=1
     gap=position_kl(donor[:,pos:],lg[:,pos:]);shift=position_kl(base[:,pos:],lg[:,pos:])
     rows.append(dict(context=ci,anchor=anchor,receiver=rec,arm=arm,control=control,
      baseline_gap_at_edit=float(baseline_gap[0]),baseline_gap_future=float(baseline_gap[1:].mean()),
      donor_gap_at_edit=float(gap[0]),donor_gap_future=float(gap[1:].mean()),
      recipient_change_at_edit=float(shift[0]),recipient_change_future=float(shift[1:].mean())))
     if control==0:outputs[arm]=lg[:,pos:].detach().cpu()
     if ci==0 and ai==0 and rec==0:
      samples.append(dict(row_index=len(rows)-1,donor=donor[:,pos:].cpu(),base=base[:,pos:].cpu(),edited=lg[:,pos:].cpu()))
    interaction=outputs['all_positions']-outputs['changed_word']-outputs['other_positions']+base[:,pos:].cpu()
    interaction-=interaction.mean(-1,keepdim=True)
    full=outputs['all_positions']-base[:,pos:].cpu();full-=full.mean(-1,keepdim=True)
    directions[-1].update(interaction_future_rms=float(interaction[:,1:].square().mean().sqrt()),full_future_rms=float(full[:,1:].square().mean().sqrt()))
   print('completed',ci,anchor,flush=True)
 assert identity==0 and before_error==0
 def summarize(rr):
  return dict(n=len(rr),mean_donor_gap={x:sum(r['donor_gap_'+x] for r in rr)/len(rr) for x in ['at_edit','future']},
   remaining_donor_gap={x:sum(r['donor_gap_'+x] for r in rr)/sum(r['baseline_gap_'+x] for r in rr) for x in ['at_edit','future']},
   mean_recipient_change={x:sum(r['recipient_change_'+x] for r in rr)/len(rr) for x in ['at_edit','future']})
 summary={a:summarize([r for r in rows if r['arm']==a]) for a in sorted(set(r['arm'] for r in rows))}
 by_context={str(ci):{a:summarize([r for r in rows if r['arm']==a and r['context']==ci]) for a in summary} for ci in range(4)}
 result=dict(status='completed_exploratory',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),panel_source_sha256=hashlib.sha256((Path(__file__).parent/'probe_mlp2_token_transfer.py').read_bytes()).hexdigest(),checkpoint_sha256=receipt.weights_sha256,
  protocol='P5 induced MLP1 response from native prefix MLP0 edit. On unedited recipient MLP0 background, transplant changed-word response, complement, or union. Four signed permutations at same changed-word position, matching that response norm. Donor distribution target; intact recipient baseline.',
  forward_count=calls,identity_error=identity,response_before_changed_word_error=before_error,summary=summary,by_context=by_context,rows=rows,directions=directions,
  limitations=['Opened small authored panel; each pair differs in one token. No broad corpus inference.','Positions are localized, not neurons, dimensions or semantic features.','All responses are generated by native model under a donor-informed intervention.','Removing a response subset tests this induced intervention, not global necessity of the corresponding native write.'])
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');torch.save(samples,'/tmp/emlp_p6_logits.pt');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

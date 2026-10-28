"""Exploratory native MLP1-write response probe; fixed synthetic prompts, no fit."""
import argparse,json,hashlib
from pathlib import Path
import torch
from inspect_model import load_model,encode,trace

PAIRS=[
 ('The dog waited beside the','The dogs waited beside the'),
 ('She gave a book to the','He gave a book to the'),
 ('The movie was wonderful and the','The movie was terrible and the'),
 ('Paris is a city and the','London is a city and the'),
 ('There was one box near the','There were two boxes near the'),
 ('Yesterday they walked past the','Today they walk past the'),
 ('The red car stopped by the','The blue car stopped by the'),
 ('Alice spoke quietly to the','Bob spoke loudly to the'),
]

def kl(a,b):
 a=a.double().log_softmax(-1); b=b.double().log_softmax(-1)
 return float((a.exp()*(a-b)).sum(-1).mean())

def main():
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--output',type=Path,required=True)
 args=ap.parse_args(); torch.set_num_threads(6); torch.set_float32_matmul_precision('highest')
 model,receipt=load_model(); device=next(model.parameters()).device
 rows=[]; forward_count=0; identity=0.; local_errors=[]
 for pair_id,pair in enumerate(PAIRS):
  states=[]
  for text in pair:
   ids=encode(text,device); logits,c=trace(model,ids); forward_count+=1
   states.append((ids,logits,c))
  assert states[0][0][0,-1]==states[1][0][0,-1]
  for receiver in (0,1):
   ids,base,c=states[receiver]; donor=states[1-receiver][2]
   delta=donor[1]['native_write'][:,-1]-c[1]['native_write'][:,-1]
   frozen=c[2]['native_write']
   no2,_=trace(model,ids,replace=lambda i,z,y:torch.zeros_like(y) if i==2 else y); forward_count+=1
   ident,_=trace(model,ids,replace=lambda i,z,y:frozen if i==2 else y);forward_count+=1
   identity=max(identity,float((base-ident).abs().max()))
   for control in range(5):
    if control==0: edit=delta; kind='context'
    else:
     g=torch.Generator(device=device).manual_seed(2026102800+pair_id*100+receiver*10+control)
     perm=torch.randperm(delta.shape[-1],device=device,generator=g)
     sign=torch.randint(0,2,delta.shape,device=device,generator=g)*2-1
     edit=delta[:,perm]*sign; kind='signed_permutation'
    norm_error=float(abs(edit.norm()-delta.norm())/delta.norm().clamp_min(1e-12))
    assert norm_error<1e-6
    out={}; response_norm=None
    for background in ('live','frozen_write','omitted'):
     def replace(i,z,y):
      if i==1:
       changed=y.clone();changed[:,-1]+=edit;return changed
      if i==2 and background=='frozen_write':return frozen
      if i==2 and background=='omitted':return torch.zeros_like(y)
      return y
     lg,cc=trace(model,ids,replace=replace);forward_count+=1
     reference=no2 if background=='omitted' else base
     out[background]=dict(final_KL=kl(reference[:,-1],lg[:,-1]),
       centered_logit_rms=float(((lg[:,-1]-reference[:,-1])-(lg[:,-1]-reference[:,-1]).mean(-1,keepdim=True)).square().mean().sqrt()))
     if background=='live':
      change=cc[2]['native_write'][:,-1]-frozen[:,-1]
      response_norm=float(change.norm())
      # Exact local state/write accounting before subsequent suffix.
      predicted=(cc[2]['pre_mlp']-c[2]['pre_mlp'])+(cc[2]['native_write']-frozen)
      actual=cc[2]['post_mlp']-c[2]['post_mlp']
      local_errors.append(float((predicted-actual).abs().max()))
    rows.append(dict(pair=pair_id,receiver=receiver,kind=kind,control=control,
      edit_norm=float(edit.norm()),norm_match_relative_error=norm_error,mlp2_response_norm=response_norm,metrics=out))
  print(f'pair {pair_id+1}/{len(PAIRS)} complete',flush=True)
 assert identity==0
 summary={}
 for kind in ('context','signed_permutation'):
  rr=[r for r in rows if r['kind']==kind]
  totals={b:sum(r['metrics'][b]['final_KL'] for r in rr) for b in ('live','frozen_write','omitted')}
  summary[kind]=dict(n=len(rr),mean_final_KL={b:v/len(rr) for b,v in totals.items()},
    pooled_live_over_frozen_KL=totals['live']/max(totals['frozen_write'],1e-30),
    live_lower_than_frozen=sum(r['metrics']['live']['final_KL']<r['metrics']['frozen_write']['final_KL'] for r in rr))
 payload=dict(status='completed_exploratory',checkpoint_sha256=receipt.weights_sha256,
  prompts=PAIRS,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  protocol='Recipient persistent x0/value bus unchanged; final-position MLP1 write delta from same-token donor. No task labels or fitting. Five directions per recipient: actual delta plus four signed permutations. All full-size edits. MLP2 write frozen at recipient baseline or omitted; omitted output compared against its own omission baseline.',
  limitations=['8 authored pairs, 16 directional edits, no representative corpus or inferential confidence claim.',
    'Frozen write is an oracle intervention, not an executable learned replacement.',
    'Omission changes the baseline distribution; compare within each background, not a common-reference KL ratio.',
    'Norm-matched permutations need not be on-manifold. Direction-specific attenuation does not alone prove semantic repair.'],
  forward_count=forward_count,identity_max_logit_error=identity,max_local_accounting_error=max(local_errors),summary=summary,rows=rows)
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

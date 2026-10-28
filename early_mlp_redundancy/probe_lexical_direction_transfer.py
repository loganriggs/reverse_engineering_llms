"""P10: lexical-pair-held-out mean-direction transfer with explicitly oracle norm."""
import argparse,hashlib,json
from pathlib import Path
import torch
from inspect_model import load_model,trace
from probe_training_task_responses import kl,margin
P=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 panelpath=P/'validation/training_source_context_audit.json';screenpath=P/'validation/preamble_competence.json'
 originals={r['id']:r for r in json.loads(panelpath.read_text())['rows'] if r['family'] in ('agreement','sentiment')}
 screen=json.loads(screenpath.read_text());passed={(r['family'],r['preamble_index']) for r in screen['cohorts'] if r['competence_gate_pass'] and r['all_single_token']}
 tests=[r for r in screen['rows'] if r['id'] in originals and (r['family'],r['preamble_index']) in passed];assert len(tests)==40
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');model,receipt=load_model();device=next(model.parameters()).device;calls=0
 def states(ids):
  nonlocal calls
  out=[]
  for x in ids:
   tokens=torch.tensor([x],device=device);lg,c=trace(model,tokens);calls+=1;out.append((tokens,lg,c))
  return out
 def response(st,rec,pos):
  nonlocal calls
  def patch(i,z,y):
   if i!=0:return y
   out=y.clone();out[:,pos]=st[1-rec][2][0]['native_write'][:,pos];return out
  _,c=trace(model,st[rec][0],replace=patch);calls+=1;return c[1]['native_write'][:,pos]-st[rec][2][1]['native_write'][:,pos]
 bank={};bankmeta={};lookup={}
 for pair in originals.values():
  pos=pair['source_position'];lex=tuple(pair['source_token_pair']);st=states(pair['token_ids'])
  for rec in (0,1):
   v=response(st,rec,pos);key=(pair['family'],lex,rec,tuple(pair['token_ids'][rec][:pos+1]),tuple(pair['token_ids'][1-rec][:pos+1]))
   if key in bank:assert torch.equal(bank[key],v)
   else:
    bank[key]=v;bankmeta[key]=dict(family=pair['family'],lexical_pair=list(lex),receiver=rec,recipient_prefix=list(key[3]),donor_prefix=list(key[4]))
   lookup[(pair['id'],rec)]=key
 bankkeys=list(bank);bankindex={k:i for i,k in enumerate(bankkeys)};folds={};foldmeta=[]
 for family in ('agreement','sentiment'):
  lexes=sorted({k[1] for k in bank if k[0]==family});assert len(lexes)==4
  for held in lexes:
   for rec in (0,1):
    groups=[];indices=[]
    for lex in lexes:
     if lex==held:continue
     keys=[k for k in bank if k[:3]==(family,lex,rec)];assert keys
     groups.append(torch.stack([bank[k]/bank[k].norm() for k in keys]).mean(0));indices.extend(bankindex[k] for k in keys)
    v=torch.stack(groups).mean(0);v=v/v.norm();folds[(family,held,rec)]=v
    foldmeta.append(dict(family=family,held_pair=list(held),receiver=rec,training_bank_indices=indices))
 rows=[];samples=[];targets=[];sampled=set()
 for pi,pair in enumerate(tests):
  orig=originals[pair['id']];lex=tuple(orig['source_token_pair']);pos=pair['changed_positions'][0];st=states(pair['token_ids'])
  for rec in (0,1):
   native=response(st,rec,pos);old=bank[lookup[(pair['id'],rec)]];direction=folds[(pair['family'],lex,rec)]
   targets.append(dict(id=pair['id'],preamble_index=pair['preamble_index'],receiver=rec,native=native.cpu(),shared=direction.cpu()))
   edits=[('native_response',native),('same_lexical_direction',old/old.norm()*native.norm()),('heldout_mean_direction',direction*native.norm())]
   for j in range(4):
    g=torch.Generator(device=device).manual_seed(2026110300+pi*100+rec*10+j);perm=torch.randperm(native.shape[-1],device=device,generator=g);sign=torch.randint(0,2,native.shape,device=device,generator=g)*2-1
    edits.append(('random',native[:,perm]*sign))
   base=st[rec][1][0,-1];teacher=st[1-rec][1][0,-1];groups=orig['answer_groups'][1-rec];variant=int(orig['variants'][1-rec])
   for kind,edit in edits:
    assert torch.allclose(edit.norm(),native.norm(),rtol=1e-6,atol=1e-6)
    def patch(i,z,y):
     if i!=1:return y
     out=y.clone();out[:,pos]+=edit;return out
    lg,_=trace(model,st[rec][0],replace=patch);calls+=1;candidate=lg[0,-1]
    rows.append(dict(id=pair['id'],family=pair['family'],lexical_pair=list(lex),preamble_index=pair['preamble_index'],receiver=rec,kind=kind,native_norm=float(native.norm()),edit_norm=float(edit.norm()),direction_cosine=float(torch.nn.functional.cosine_similarity(edit,native).item()),native_donor_KL=kl(teacher,base),remaining_donor_KL=kl(teacher,candidate),donor_margin_gap=margin(teacher,groups,variant)-margin(base,groups,variant),margin_gain=margin(candidate,groups,variant)-margin(base,groups,variant)))
    samplekey=(pair['family'],lex,pair['preamble_index'],kind)
    if rec==0 and kind=='heldout_mean_direction' and samplekey not in sampled:
     sampled.add(samplekey);samples.append(dict(row=len(rows)-1,teacher=teacher.cpu(),baseline=base.cpu(),candidate=candidate.cpu(),groups=groups,variant=variant))
  print('completed',pi+1,len(tests),flush=True)
 summaries=[]
 for family,pre in sorted({(r['family'],r['preamble_index']) for r in rows}):
  for lex in [None]+sorted({tuple(r['lexical_pair']) for r in rows if r['family']==family}):
   arms={}
   for kind in ('native_response','same_lexical_direction','heldout_mean_direction','random'):
    rr=[r for r in rows if r['family']==family and r['preamble_index']==pre and r['kind']==kind and (lex is None or tuple(r['lexical_pair'])==lex)]
    arms[kind]=dict(n=len(rr),KL_recovery=1-sum(r['remaining_donor_KL'] for r in rr)/sum(r['native_donor_KL'] for r in rr),margin_recovery=sum(r['margin_gain'] for r in rr)/sum(r['donor_margin_gap'] for r in rr),mean_cosine=sum(r['direction_cosine'] for r in rr)/len(rr))
   summaries.append(dict(family=family,preamble_index=pre,lexical_pair=lex,arms=arms))
 out=dict(status='completed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),panel_sha256=hashlib.sha256(panelpath.read_bytes()).hexdigest(),competence_sha256=hashlib.sha256(screenpath.read_bytes()).hexdigest(),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,bank=[bankmeta[k] for k in bankkeys],folds=foldmeta,summaries=summaries,rows=rows,protocol='P9 fixed lexical cross-validation: original-context mean unit directions from three other lexical pairs, exact source-prefix deduplication, equal lexical weights, separate receiver directions. Test only P8 competent prefixed agreement/sentiment cohorts. All arms use oracle native response norm. Previously inspected training data, not untouched tasks. Labels choose family/orientation, not gain or output optimization.')
 args.output.write_text(json.dumps(out,indent=2)+'\n');torch.save(dict(bank=[bank[k].cpu() for k in bankkeys],targets=targets,samples=samples),'/tmp/emlp_p10_samples.pt')
 print(json.dumps([r for r in summaries if r['lexical_pair'] is None],indent=2))
if __name__=='__main__':main()

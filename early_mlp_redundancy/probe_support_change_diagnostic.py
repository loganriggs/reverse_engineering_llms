"""M64: fixed native responses to within-support and support-switch changes."""
from pathlib import Path
import json,hashlib,torch
from inspect_model import load_model,trace
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def margin(z,g,v):return float(torch.logsumexp(z.double()[g[v]],0)-torch.logsumexp(z.double()[g[1-v]],0))
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);pp=V/'support_change_diagnostic_protocol.json';gp=V/'support_change_geometry.json';geometry=json.loads(gp.read_text());assert geometry['status']=='passed' and geometry['protocol_sha256']==sha(pp) and geometry['source_sha256']==sha(P/'prepare_support_change_diagnostic.py');native=json.loads((V/'source_holdout_native.json').read_text());sp=Path(native['samples_path']);assert sha(sp)==native['samples_sha256'];saved=torch.load(sp,weights_only=True);pairs=json.loads((V/'source_holdout_native_protocol.json').read_text())['pairs'];old=json.loads((V/'state_direction_factorial.json').read_text());prior={(x['kind'],x['seed'],x['arm'],x['key'],x['region']):x for x in old['rows'] if x['corner']=='native_reconstructed'};targets={x['sample_key']:x for x in native['records']};op=V/'support_change_responses.json';assert not op.exists();dest=A/'emlp_m64_responses';dest.mkdir(exist_ok=True);progress=A/'emlp_m64_progress.pt';rows=[];files=[];calls=0;completed=0;replays=0;model,receipt=load_model()
 if progress.exists():
  p=torch.load(progress,weights_only=True);assert p['source_sha256']==sha(Path(__file__)) and p['geometry_sha256']==sha(gp) and p['protocol_sha256']==sha(pp);rows,files,calls,completed,replays=[p[k] for k in ['rows','files','calls','completed','replays']]
 for pi,pair in enumerate(pairs):
  if pi<completed:continue
  f=next(x for x in geometry['files'] if x['pair_id']==pair['id']);path=Path(f['path']);assert sha(path)==f['sha256'];blob=torch.load(path,weights_only=True);changed={};bases=[]
  for ep in [0,1]:
   ids=torch.tensor([pair['token_ids'][ep]],device='cuda');out,cap=trace(model,ids);calls+=1;key=pair['id']+f'|{ep}';assert torch.equal(out[0,-1].cpu(),saved['baselines'][key]['logits']);bases.append(out[0,-1])
  for entry in blob['entries']:
   ident=entry['identity'];ep=int(ident['key'].rsplit('|',1)[1]);ids=torch.tensor([pair['token_ids'][ep]],device='cuda');base=bases[ep];nlp=base.double().log_softmax(-1);tk=ident['key']+'|1|'+ident['region'];target=targets[tk];teacher=saved['changed'][tk].cuda().double().log_softmax(-1)-nlp;g=pair['answer_groups'][ep];variant=pair['variants'][ep];ms=':'.join(str(ident[k]) for k in ['kind','seed','arm'])
   for term in ['within','switch','full']:
    delta=torch.zeros((1,ids.shape[1],1152),device='cuda');delta[:,entry['positions']]=entry['deltas'][term].cuda()
    if not bool(delta.count_nonzero()):z=base
    else:out,_=trace(model,ids,replace=lambda j,x,w:w+delta if j==1 else w);calls+=1;z=out[0,-1]
    error=float((nlp.exp()*(z.double().log_softmax(-1)-nlp-teacher).square()).sum());me=(margin(z,g,variant)-margin(base,g,variant)-target['margin_change'])**2;row=ident|dict(term=term,response_error=error,target_energy=target['response_energy'],margin_error=me,margin_target_energy=target['margin_change']**2);rows.append(row);changed[ms+f'|{ep}|'+ident['region']+'|'+term]=z.cpu()
    if term=='full':
     prev=prior[ident['kind'],ident['seed'],ident['arm'],ident['key'],ident['region']];assert error==prev['response_error'] and me==prev['margin_error'];replays+=1
  destpath=dest/path.name;torch.save(dict(pair_id=pair['id'],changed=changed),destpath);files.append(dict(pair_id=pair['id'],path=str(destpath),sha256=sha(destpath)));tmp=progress.with_suffix('.tmp');torch.save(dict(source_sha256=sha(Path(__file__)),geometry_sha256=sha(gp),protocol_sha256=sha(pp),rows=rows,files=files,calls=calls,completed=pi+1,replays=replays),tmp);tmp.replace(progress);print('PAIRS',pi+1,'FORWARDS',calls,flush=True)
 assert len(rows)==4536 and replays==1512
 op.write_text(json.dumps(dict(status='completed',source_sha256=sha(Path(__file__)),protocol_sha256=sha(pp),geometry_sha256=sha(gp),native_sha256=sha(V/'source_holdout_native.json'),rows=rows,files=files,forward_count=calls,full_corner_replays=replays),indent=2)+'\n');print('DONE',calls,flush=True)
if __name__=='__main__':main()

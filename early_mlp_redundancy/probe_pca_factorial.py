"""M73 PCA native/reconstructed background crossed with native/reconstructed delta."""
from pathlib import Path
import hashlib,json,torch
from inspect_model import load_model,trace
from pca_encoder import PCAEncoder
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def margin(z,g,v):return float(torch.logsumexp(z.double()[g[v]],0)-torch.logsumexp(z.double()[g[1-v]],0))
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);pp=V/'pca_factorial_execution.json';q=json.loads(pp.read_text());mp=V/'pca_evaluation_models.json';np=V/'source_holdout_native.json';npp=V/'source_holdout_native_protocol.json';xp=V/'least_squares_factorial.json'
 for path,key in [(mp,'models_sha256'),(np,'holdout_native_results_sha256'),(npp,'holdout_native_protocol_sha256'),(xp,'existing_corner_sha256')]:assert sha(path)==q[key]
 assert q['pca_protocol_sha256']==sha(V/'pca_encoder_baseline_protocol.json');manifest=json.loads(mp.read_text());assert manifest['encoder_source_sha256']==sha(P/'pca_encoder.py');models=json.loads(mp.read_text())['models'];native=json.loads(np.read_text());pairs=json.loads(npp.read_text())['pairs'];old=json.loads(xp.read_text());sp=Path(native['samples_path']);assert sha(sp)==native['samples_sha256'];saved=torch.load(sp,weights_only=True);prior={(r['key'],r['kind'],r['seed'],r['arm'],r['region'],r['corner']):r for r in old['rows']};model,receipt=load_model();mlp=model.transformer.h[1].mlp;states={}
 for e in models:states[e['kind'],e['seed'],e['arm']]=PCAEncoder(e['rank'])
 op=V/'pca_factorial.json';assert not op.exists();dest=A/'emlp_m73_factorial_samples';dest.mkdir(exist_ok=True);progress=A/'emlp_m73_factorial_progress.pt';rows=[];geometry=[];files=[];calls=0;completed=0;replays=0
 if progress.exists():
  p=torch.load(progress,weights_only=True);assert p['protocol_sha256']==sha(pp) and p['source_sha256']==sha(Path(__file__));rows,geometry,files,calls,completed,replays=[p[k] for k in ['rows','geometry','files','calls','completed','replays']]
 for pi,pair in enumerate(pairs):
  if pi<completed:continue
  cs=[];blob=dict(pair_id=pair['id'],writes=[],reconstructions={},baselines={},changed={})
  for ep in [0,1]:
   ids=torch.tensor([pair['token_ids'][ep]],device='cuda');out,cap=trace(model,ids);calls+=1;key=pair['id']+f'|{ep}';base=out[0,-1];assert torch.equal(base.cpu(),saved['baselines'][key]['logits']);w=mlp.Down(cap[1]['products']);cs.append(dict(ids=ids,key=key,base=base,write=w));blob['writes'].append(w.cpu())
  for e in models:
   mk=e['kind'],e['seed'],e['arm'];ms=':'.join(map(str,mk));enc=states[mk];rec=[];bases=[]
   for ep,c in enumerate(cs):
    r,support,code,info=enc.reconstruct(c['write'],e['variant']);rec.append(r);out,_=trace(model,c['ids'],replace=lambda j,z,w:r+mlp.Down_bias if j==1 else w);calls+=1;bases.append(out[0,-1]);blob['baselines'][ms+f'|{ep}']=out[0,-1].cpu();blob['reconstructions'][ms+f'|{ep}']=dict(write=r.cpu(),code=code.cpu(),support=support.cpu(),info=info)
   for ep,c in enumerate(cs):
    key=c['key'];g=pair['answer_groups'][ep];variant=pair['variants'][ep];nlp=c['base'].double().log_softmax(-1)
    for target in [r for r in native['records'] if r['pair_id']==pair['id'] and r['endpoint']==ep]:
     assert target['status']=='evaluated';region=target['region'];positions=target['recipient_positions'];donorpos=target['donor_positions'];nd=torch.zeros_like(c['write']);nd[:,positions]=saved['deltas'][target['sample_key']]['delta'].cuda();rd=torch.zeros_like(nd);rd[:,positions]=rec[1-ep][:,donorpos]-rec[ep][:,positions];assert torch.equal(nd[:,positions],cs[1-ep]['write'][:,donorpos]-c['write'][:,positions]);teacher=saved['changed'][target['sample_key']].cuda();response=teacher.double().log_softmax(-1)-nlp
     ident={k:e[k] for k in ['kind','seed','arm']}|dict(key=key,family=pair['family'],region=region);energy=float(nd.double().square().sum());error=float((rd.double()-nd.double()).square().sum());geometry.append(ident|dict(target_delta_energy=energy,delta_error=error,relative_delta_error=error/energy if energy else None,predicted_delta_energy=float(rd.double().square().sum())))
     for corner,delta,background in [('reconstructed_native',nd,rec[ep]),('native_reconstructed',rd,c['write']),('reconstructed_reconstructed',rd,rec[ep])]:
      base=c['base'] if corner=='native_reconstructed' else bases[ep]
      if not bool(delta.count_nonzero()):changed=base
      else:
       if corner=='native_reconstructed':out,_=trace(model,c['ids'],replace=lambda j,z,w:w+delta if j==1 else w)
       else:out,_=trace(model,c['ids'],replace=lambda j,z,w:background+delta+mlp.Down_bias if j==1 else w)
       calls+=1;changed=out[0,-1]
      sr=changed.double().log_softmax(-1)-base.double().log_softmax(-1);er=float((nlp.exp()*(sr-response).square()).sum());effect=margin(changed,g,variant)-margin(base,g,variant);row=ident|dict(corner=corner,response_error=er,target_energy=target['response_energy'],margin_error=(effect-target['margin_change'])**2,margin_target_energy=target['margin_change']**2);rows.append(row);blob['changed'][ms+f'|{ep}|'+region+'|'+corner]=changed.cpu()
      if (key,e['kind'],e['seed'],e['arm'],region,corner) in prior:
       previous=prior[key,e['kind'],e['seed'],e['arm'],region,corner];assert er==previous['response_error'] and row['margin_error']==previous['margin_error'];replays+=1
  path=dest/(hashlib.sha256(pair['id'].encode()).hexdigest()+'.pt');torch.save(blob,path);files.append(dict(pair_id=pair['id'],path=str(path),sha256=sha(path)));tmp=progress.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),rows=rows,geometry=geometry,files=files,calls=calls,completed=pi+1,replays=replays),tmp);tmp.replace(progress);print('PAIRS',pi+1,'FORWARDS',calls,flush=True)
 summaries=[]
 for e in models:
  for region in ['changed_input_positions','final_position','identity']:
   for corner in ['reconstructed_native','native_reconstructed','reconstructed_reconstructed']:
    for family in [None]+sorted({p['family'] for p in pairs}):
     rr=[r for r in rows if all(r[k]==e[k] for k in ['kind','seed','arm']) and r['region']==region and r['corner']==corner and (family is None or r['family']==family)];en=sum(r['target_energy'] for r in rr);er=sum(r['response_error'] for r in rr);me=sum(r['margin_target_energy'] for r in rr);mr=sum(r['margin_error'] for r in rr);summaries.append({k:e[k] for k in ['kind','seed','arm']}|dict(region=region,corner=corner,family=family,n=len(rr),target_energy=en,error_energy=er,relative_response_error=er/en if en else None,margin_target_energy=me,margin_error_energy=mr,relative_margin_error=mr/me if me else None))
 assert len(rows)==504 and len(geometry)==168 and replays==0 and calls==420
 op.write_text(json.dumps(dict(status='completed',protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),models_sha256=sha(mp),forward_count=calls,old_corner_replays=replays,rows=rows,geometry=geometry,summaries=summaries,files=files,scope='Native/native teacher corner has zero prediction error by definition and is not repeated per candidate; other three corners retain all identities. No additive attribution assumed.'),indent=2)+'\n');print('DONE',calls,flush=True)
if __name__=='__main__':main()

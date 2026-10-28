"""H597: locked-basis evaluation on fresh lexical and wording combinations."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure,groups,continuation_id
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_number_fresh_transfer'
def suite():
 regular=[('atom','atoms'),('planet','planets'),('molecule','molecules'),('crystal','crystals'),('flower','flowers'),('building','buildings'),('vehicle','vehicles'),('device','devices')]
 irregular=[('ox','oxen'),('louse','lice'),('cactus','cacti'),('fungus','fungi'),('nucleus','nuclei'),('analysis','analyses'),('criterion','criteria'),('phenomenon','phenomena')]
 attractors=[('researcher','researchers'),('student','students'),('inspector','inspectors'),('assistant','assistants'),('ranger','rangers'),('architect','architects'),('driver','drivers'),('technician','technicians')];rows=[]
 for kind,ss in [('regular',regular),('irregular',irregular)]:
  for i,(subject,attractor) in enumerate(zip(ss,attractors)):
   for wording in ('old','new'):
    for frame in ('preposition','fronted','relative'):
     for sn in (0,1):
      for an in (0,1):
       prep='beside' if wording=='old' else 'across from';clause='can see' if wording=='old' else 'might notice'
       text=f'The {subject[sn]} {prep} the {attractor[an]} usually' if frame=='preposition' else f'{prep.capitalize()} the {attractor[an]}, the {subject[sn]} usually' if frame=='fronted' else f'The {subject[sn]} that the {attractor[an]} {clause} usually'
       rows.append(dict(id=f'fresh-{kind}-{i}-{frame}-{wording}-{sn}-{an}',kind=kind,item=i,frame=frame+'_'+wording,wording=wording,subject_number=sn,attractor_number=an,prompt=text))
 return rows
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_number_context_coverage.json';bp=ART/'bilin18_agreement_write_subspace_locked_bases.pt';b=torch.load(bp,weights_only=False);axis=b['pca'][0]*torch.sign(b['pca'][0]@b['mean']);scale=float(axis@b['mean']);doses=[.0625];ranks=[32,64];basispath=ART/'bilin18_number_context_coverage_locked_bases.pt';frozen=torch.load(basispath,weights_only=False);spaces=frozen['spaces'];cs=frozen['context_singular_values'];train=frozen['training_ids'];bh=hashlib.sha256(basispath.read_bytes()).hexdigest();containment=float((spaces['old32']-(spaces['old32']@spaces['new64'].T)@spaces['new64']).norm()/spaces['old32'].norm())
 cfg=dict(experiment='H597',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),number_basis_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),context_basis_sha256=bh,doses=doses,design='Freeze H596 old32/new32/new64 spaces and H584 number direction. New regular and irregular subjects plus new attractors. Cross familiar wording (beside/can see) and new wording (across from/might notice), both number factors and three frames. Match each recipient to unfronted anchor within wording. Full context switch, unchanged, projection and response oracle. No refit or filtering.',gate='Per type/frame and attractor number: >=4 pairs, >=75% both-endpoint candidate correctness, >=50% intended-group full-vocabulary top1. Report all cells; failures remain in interventions.',scope='Fresh combinations relative to enumerated preceding manifests, not model-training novelty. Some nouns/contexts may be less natural. Oracle remains target-informed. New wording changes more than one syntactic/positional property; same-wording cells isolate lexical transfer more closely.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));rows=suite();mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));mh=hashlib.sha256(mp.read_bytes()).hexdigest();oldtexts=set()
 for stem in ['bilin18_number_subject_attractor','bilin18_number_fronted_attractor','bilin18_task_boundary_factorial']:
  for row in json.loads((ART/f'{stem}_manifest.json').read_text()):oldtexts.update([row['prompt']] if 'prompt' in row else row['prompts'])
 overlap=sum(row['prompt'] in oldtexts for row in rows);assert overlap==0;tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);all_ex={}
 for row in rows:
  gs=[[continuation_id(tok,row['prompt'],t) for t in g] for g in groups('agreement',['is','are'])];assert all(all(t is not None for t in g) for g in gs);all_ex[row['id']]=dict(row,ids=tok.encode(row['prompt'],add_special_tokens=False),groups=gs)
 ex={k:e for k,e in all_ex.items() if not e['frame'].startswith('preposition')};idx={(e['kind'],e['item'],e['subject_number'],e['attractor_number'],e['frame']):k for k,e in all_ex.items()};pk={k:idx[e['kind'],e['item'],e['subject_number'],e['attractor_number'],'preposition_'+e['wording']] for k,e in ex.items()}
 model=load_full_model('cuda');assert not model.config.gated
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 mlp=model.blocks[11].mlp;inputs={};writes={};captured_logits={};native_axis={};batches=defaultdict(list)
 for k,e in all_ex.items():batches[len(e['ids'])].append(k)
 print('LOCKED',len(rows),'fresh prompts',flush=True)
 for length,kk in batches.items():
  for st in range(0,len(kk),24):
   bb=kk[st:st+24];ids=torch.tensor([all_ex[k]['ids'] for k in bb],device='cuda')
   for dose in [0.,-.0625,.0625]:
    cap={}
    def pre(m,args):cap['input']=args[0][:,-1].cpu()
    def post(m,args,out):cap['write']=out[:,-1].cpu()
    def ah(m,args,out):cap['axis']=(out[0][:,-1]@axis.cuda()).cpu()
    handles=[mlp.register_forward_pre_hook(pre),mlp.register_forward_hook(post),model.blocks[5].attn.register_forward_hook(ah)]
    try:z,_=run(model,ids,patch=None if dose==0 else (5,(axis*scale*dose).expand(len(bb),-1)))
    finally:
     for h in handles:h.remove()
    for ix,k in enumerate(bb):
     inputs[k,dose]=cap['input'][ix].clone();writes[k,dose]=cap['write'][ix].clone();captured_logits[k,dose]=z[ix].cpu()
     if dose==0:native_axis[k]=float(cap['axis'][ix])
 competence={};native_records=[]
 for k,e in all_ex.items():native_records.append(dict(id=k,kind=e['kind'],frame=e['frame'],subject_number=e['subject_number'],attractor_number=e['attractor_number'],length=len(e['ids']),axis_coefficient=native_axis[k],**measure(captured_logits[k,0.],e['groups'])))
 for kind in ('regular','irregular'):
  for frame in sorted({e['frame'] for e in all_ex.values()}):
   for an in (0,1):
    ids=[idx[kind,i,sn,an,frame] for i in range(8) for sn in (0,1)];mm={k:measure(captured_logits[k,0.],all_ex[k]['groups']) for k in ids};both=float(np.mean([mm[idx[kind,i,0,an,frame]]['margin']>0 and mm[idx[kind,i,1,an,frame]]['margin']<0 for i in range(8)]));top=float(np.mean([mm[k]['top1'][all_ex[k]['subject_number']] for k in ids]));competence[f'{kind}_{frame}_attractor{an}']=dict(pairs=8,both_endpoint_choice=both,full_vocab_top1=top,passes=both>=.75 and top>=.5)
 print('COMPETENCE',json.dumps(competence),flush=True);saved={'hybrids':{}}
 def response(z,e):
  z=z.cuda();e=e.cuda();return mlp.Down(mlp.Left(z)*mlp.Right(e)+mlp.Left(e)*mlp.Right(z)+mlp.Left(e)*mlp.Right(e)).cpu()
 for k in ex:
  for dose in [-.0625,.0625]:
   de=inputs[k,dose]-inputs[k,0.];saved['hybrids'][k,dose,'recipient_recipient']=response(inputs[k,0.],de);saved['hybrids'][k,dose,'preposition_recipient']=response(inputs[pk[k],0.],de)
 torch.save(dict(examples=all_ex,inputs=inputs,writes=writes,logits=captured_logits,native_axis=native_axis,hybrids=saved['hybrids']),ART/f'{STEM}_captures.pt')
 def T(c,e):
  c=c.cuda();e=e.cuda();return mlp.Down(mlp.Left(c)*mlp.Right(e)+mlp.Left(e)*mlp.Right(c)).cpu()
 vecs={};buckets=defaultdict(list);identityerrors=[];normerrors=[];modes=['full','native']+[f'{space}_{method}' for space in spaces for method in ['projection','oracle']];oracle_geometry=[];coefficients={}
 for k,e in ex.items():
  dc=inputs[pk[k],0.]-inputs[k,0.]
  for dose in [sgn*d for d in doses for sgn in (-1,1)]:
   de=inputs[k,dose]-inputs[k,0.];full=T(dc,de);vv={'full':full,'native':torch.zeros_like(full)}
   for space,A in spaces.items():
    projected=A.T@(A@dc);vv[space+'_projection']=T(projected,de)
    matrix=T(A,de).T.double();target=full.double();U,S,Vh=torch.linalg.svd(matrix,full_matrices=False);keep=S>S[0]*1e-10;coef=Vh[keep].T@((U[:,keep].T@target)/S[keep]);pred=matrix@coef;res=pred-target;vv[space+'_oracle']=pred.float();coefficients[k,dose,space]=coef
    mapped=T((A.double().T@coef).float(),de).double();ortho=float((matrix.T@res).norm()/(matrix.norm()*target.norm()));oracle_geometry.append(dict(id=k,kind=e['kind'],frame=e['frame'],dose=dose,space=space,rank=int(keep.sum()),condition=float(S[0]/S[-1]),normal_equation_relative_error=ortho,coefficient_norm_ratio=float(coef.norm()/dc.double().norm()),projection_response_error=float((vv[space+'_projection'].double()-target).norm()/target.norm()),oracle_response_error=float(res.norm()/target.norm()),coefficient_replay_relative_error=float((mapped-pred).norm()/target.norm())))
   actual=saved['hybrids'][k,dose,'preposition_recipient']-saved['hybrids'][k,dose,'recipient_recipient'];identityerrors.append(float((actual-full).norm()/full.norm()))
   for mode,v in vv.items():vecs[k,dose,mode]=v
   buckets[len(e['ids'])].append((k,dose))
 print('LOCKED',len(vecs),'interventions',flush=True);refs={};result=dict(config=cfg,competence=competence,native=native_records,manifest_sha256=mh,basis_sha256=bh,discovery_energy={str(r):float(cs[:r].square().sum()/cs.square().sum()) for r in ranks},oracle_geometry=oracle_geometry,interventions=[],summary={},status='running')
 for mode in modes:
  for length,jj in buckets.items():
   for st in range(0,len(jj),24):
    bb=jj[st:st+24];don=torch.stack([writes[k,0.]+saved['hybrids'][k,dose,'recipient_recipient']+vecs[k,dose,mode] for k,dose in bb]).cuda()
    def h(m,args,out):out=out.clone();out[:,-1]=don.to(out);return out
    handle=mlp.register_forward_hook(h)
    try:zz,_=run(model,torch.tensor([ex[k]['ids'] for k,d in bb],device='cuda'),patch=(5,torch.stack([axis*scale*dose for k,dose in bb])))
    finally:handle.remove()
    for ix,(k,dose) in enumerate(bb):
     z=zz[ix].cpu();e=ex[k];gs=e['groups'];ans=gs[e['subject_number']]
     if mode in ('full','native'):refs[k,dose,mode]=z
     ref=refs[k,dose,'full'];lp=z.double().log_softmax(0);rp=ref.double().log_softmax(0);margin=measure(z,gs)['margin'];rm=measure(ref,gs)['margin'];f=vecs[k,dose,'full'];result['interventions'].append(dict(id=k,kind=e['kind'],frame=e['frame'],dose=dose,mode=mode,margin=margin,margin_error=margin-rm,reference_kl=float((rp.exp()*(rp-lp)).sum()),group_nll_error=float(torch.logsumexp(rp[ans],0)-torch.logsumexp(lp[ans],0)),internal_response_error=float((vecs[k,dose,mode]-f).norm()/f.norm())))
  print('COMPLETE',mode,flush=True)
 for kind in ('regular','irregular'):
  for frame in ('fronted_old','relative_old','fronted_new','relative_new'):
   for dose in doses:
    cc=[c for c in result['interventions'] if c['kind']==kind and c['frame']==frame and abs(c['dose'])==dose];base=[c for c in cc if c['mode']=='native'];md=np.mean([abs(c['margin_error']) for c in base]);kd=np.mean([c['reference_kl'] for c in base])
    for mode in modes:
     vv=[c for c in cc if c['mode']==mode];result['summary'][f'{kind}_{frame}_{dose}_{mode}']=dict(internal_response_error=float(np.mean([c['internal_response_error'] for c in vv])),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/md),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),group_nll_abs_error=float(np.mean([abs(c['group_nll_error']) for c in vv])))
 replay=[abs(c['margin']-measure(captured_logits[c['id'],c['dose']],ex[c['id']]['groups'])['margin']) for c in result['interventions'] if c['mode']=='native']
 result['audits']=dict(old_span_containment_relative_error=containment,basis_refit=False,fresh_prompt_count=len(all_ex),prior_manifest_overlap=overlap,manifest_unchanged=mh==hashlib.sha256(mp.read_bytes()).hexdigest(),oracle_normal_equation_max_relative_error=max(g['normal_equation_relative_error'] for g in oracle_geometry),oracle_coefficient_replay_max_relative_error=max(g['coefficient_replay_relative_error'] for g in oracle_geometry),oracle_worse_than_projection_count=sum(g['oracle_response_error']>g['projection_response_error']+1e-6 for g in oracle_geometry),context_switch_identity_max_relative_error=max(identityerrors),matched_input_norm_max_relative_error=max(normerrors,default=0.),reference_replay_max_margin_error=max(replay),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ('margin_error','reference_kl','internal_response_error')),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),basis_unchanged=bh==hashlib.sha256(basispath.read_bytes()).hexdigest(),number_basis_unchanged=cfg['number_basis_sha256']==hashlib.sha256(bp.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));torch.save(dict(vectors=vecs,coefficients=coefficients,examples=ex,oracle_geometry=oracle_geometry),ART/f'{STEM}_oracle.pt');print(result['audits'],flush=True)
if __name__=='__main__':main()

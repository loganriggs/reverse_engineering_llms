"""H713: frozen coactivation groups, small norm-capped interventions and controls."""
import hashlib,json,time
from pathlib import Path
from collections import defaultdict
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
STEM='bilin18_coactivation_group_swaps'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

@torch.no_grad()
def metrics(base, donor, edited, nextid):
 base=base.double();bp=base.log_softmax(-1);dp=donor.double().log_softmax(-1);ep=edited.double().log_softmax(-1)
 centered=base-base.max();target=(ep.exp()*centered).sum();lo,hi=.05,20.
 for _ in range(50):
  mid=(lo+hi)/2
  if float(((centered*mid).softmax(-1)*centered).sum())<float(target):lo=mid
  else:hi=mid
 beta=(lo+hi)/2;tp=(centered*beta).log_softmax(-1)
 dist=(dp.exp()*(dp-bp)).sum();gain=dist-(dp.exp()*(dp-ep)).sum();tgain=dist-(dp.exp()*(dp-tp)).sum()
 eb=(ep.exp()*(ep-bp)).sum();et=(ep.exp()*(ep-tp)).sum();assert et<=eb+1e-7
 return dict(donor_kl_baseline=float(dist),donor_kl_gain=float(gain),temperature_donor_kl_gain=float(tgain),gain_minus_temperature=float(gain-tgain),receiver_kl=float((bp.exp()*(bp-ep)).sum()),edit_to_base_kl=float(eb),edit_to_temperature_kl=float(et),beta=beta,observed_next_token_id=nextid,observed_next_token_ce_change=float(bp[nextid]-ep[nextid]),native_entropy=float(-(bp.exp()*bp).sum()),edited_entropy=float(-(ep.exp()*ep).sum()))

@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 paths={'data':ART/'bilin18_coactivation_replication_captures.pt','groups':ART/'bilin18_rotation_coactivation_groups.json','spectra':ART/'bilin18_early_rotation_spectra_captures.pt','rotations':ART/'bilin18_early_sparse_rotations_captures.pt','model':SNAPSHOT/'pytorch_model.bin','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py','forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','runner':Path(__file__)}
 hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);sp=torch.load(paths['spectra'],weights_only=False)['states']['residual1'];Q=torch.load(paths['rotations'],weights_only=False)['rotations']['residual1','learned',7061].cuda();groups=json.loads(paths['groups'].read_text())['records'];mu=sp['mean'].float().cuda();V=sp['eigenvectors'][:,:256].float().cuda();sd=sp['eigenvalues'][:256].float().cuda().sqrt();raw=data['activations']['residual1'].reshape(-1,1152).cuda();D=(V*sd)@Q;norms=D.norm(dim=0);D=D/norms;code=(((raw-mu)@V)/sd)@Q*norms;mask=torch.zeros_like(code,dtype=torch.bool).scatter_(1,code.abs().topk(32,dim=-1).indices,True);tokens=data['token_ids'].flatten();rng=torch.Generator(device='cuda').manual_seed(713);R=torch.linalg.qr(torch.randn(1152,8,device='cuda',generator=rng)).Q;P=V[:,:8];pairs=[];patches=[];coverage=[];used_docs=set();ortherrors=[]
 for group in groups:
  assert group['site']=='residual1';anchor=group['anchor'];members=group['members'];assert len(set(members))==8
  G=torch.linalg.qr(D[:,members]).Q;ortherrors.append(float((G.T@G-torch.eye(8,device='cuda')).abs().max()));strata=defaultdict(list)
  for k in mask[:,anchor].nonzero().flatten().cpu().tolist():
   if int(data['positions'][k%16])+1<data['all_token_ids'].shape[1]:strata[int(tokens[k]),k%16].append(k)
  candidates=[]
  for entries in strata.values():
   entries=sorted(entries,key=lambda k:hashlib.sha256(f'H713:{anchor}:{k}'.encode()).hexdigest())
   candidates.extend(zip(entries[::2],entries[1::2]))
  candidates=sorted(candidates,key=lambda ab:hashlib.sha256(f'H713:{anchor}:{ab}'.encode()).hexdigest());chosen=[]
  for a,b in candidates:
   if a//16==b//16 or a//16 in used_docs or b//16 in used_docs:continue
   chosen.append((a,b));used_docs.update([a//16,b//16])
   if len(chosen)==4:break
  assert len(chosen)==4;coverage.append(dict(anchor=anchor,unordered_pairs=len(chosen),members=members))
  for a,b in chosen:
   pairid=len(pairs)//2
   for receiver,donor in [(a,b),(b,a)]:
    delta=raw[donor]-raw[receiver];component=D[:,members]@(code[donor,members]-code[receiver,members]);n=component.norm();assert n>0
    patch={'coordinate':component}
    for mode,basis in [('group_projection',G),('pca',P),('random',R)]:
     v=basis@(basis.T@delta);assert v.norm()>1e-8;patch[mode]=v*(n/v.norm())
    patches.append({k:v.cpu() for k,v in patch.items()});pairs.append(dict(anchor=anchor,members=members,pair_id=pairid,receiver=receiver,donor=donor,position=int(data['positions'][receiver%16]),token_id=int(tokens[receiver]),natural_coordinate_norm=float(n),receiver_residual_norm=float(raw[receiver].norm()),receiver_active_members=int(mask[receiver,members].sum()),donor_active_members=int(mask[donor,members].sum())))
 cfg=dict(experiment='H713',hashes=hashes,doses=[0.,.01,.025,.05],design='All8frozenH711groups onH712panel. Four hash-selected unordered pairs/group with anchor Top32active atboth endpoints, same token/position, distinctdocuments acrossallpairs; both directions. Coordinate edit sums all8rawdualcoordinates, noTop32mask in decoding. Cap editnorm at1/2.5/5percent receiverresidualnorm and at fullnaturalgroupdelta, never amplify it. Controls: rank8globalPCA, fixedrandom, and samegroupEuclideanprojection of donor delta, rescaled toidenticalnorm. Zero-dosecoordinate replay. Fittemperature toediteddistribution only. ReportdonorKL andobservednexttokenCE separately.',limitations='Small exploratory researchpanel, nottaskcorrectness. Correlationgroups are overlapping neighborhoods, notlearned gates/manifolds. One randomsubspace only; globalPCA/randombases notmatched togroup occupancy. Temperature descriptive notcausal decomposition. Pairselection requiresonlyanchoractive and may mix regimes. Originalreservedtestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 endpoints=sorted({p[k] for p in pairs for k in ['receiver','donor']});native={};replay=[];forwards=0
 for endpoint in endpoints:
  doc,pos=endpoint//16,int(data['positions'][endpoint%16]);capt={}
  def get(m,args,out):capt['x']=out[0][0,-1].cpu()
  h=model.blocks[1].register_forward_hook(get)
  try:logits,_=run(model,data['all_token_ids'][doc:doc+1,:pos+1].cuda());native[endpoint]=logits[0].cpu()
  finally:h.remove()
  replay.append(float((capt['x']-raw[endpoint].cpu()).norm()/raw[endpoint].norm().cpu()));forwards+=1
 assert max(replay)<1e-4;print('NATIVE',len(native),flush=True);records=[];logitrows=[];buckets=defaultdict(list);normerrors=[]
 for k,p in enumerate(pairs):buckets[p['position']].append(k)
 for mode in ['coordinate','group_projection','pca','random']:
  for dose in cfg['doses']:
   if dose==0 and mode!='coordinate':continue
   for pos,inds in buckets.items():
    for st in range(0,len(inds),8):
     bb=inds[st:st+8];scales=[min(1.,dose*pairs[k]['receiver_residual_norm']/pairs[k]['natural_coordinate_norm']) for k in bb];patch=torch.stack([patches[k][mode]*s for k,s in zip(bb,scales)])
     def edit(m,args,out):
      x,bus=out;x=x.clone();x[:,-1]+=patch.to(x);return x,bus
     h=model.blocks[1].register_forward_hook(edit)
     try:logits,_=run(model,torch.stack([data['all_token_ids'][pairs[k]['receiver']//16,:pos+1] for k in bb]).cuda())
     finally:h.remove()
     forwards+=len(bb)
     for n,k in enumerate(bb):
      p=pairs[k];actual=float(patch[n].norm());expected=min(p['natural_coordinate_norm'],dose*p['receiver_residual_norm']);normerrors.append(abs(actual-expected)/max(expected,1e-10));nextid=int(data['all_token_ids'][p['receiver']//16,pos+1]);r=dict(**p,mode=mode,dose=dose,edit_norm=actual,relative_edit_norm=actual/p['receiver_residual_norm'],**metrics(native[p['receiver']].cuda(),native[p['donor']].cuda(),logits[n],nextid));records.append(r);logitrows.append(logits[n].cpu())
   print('DONE',mode,dose,flush=True)
   (ART/f'{STEM}.partial.json').write_text(json.dumps(dict(status='running',records=records),indent=2))
 zero=max(r['receiver_kl'] for r in records if r['dose']==0);audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor([r['donor_kl_gain'],r['observed_next_token_ce_change'],r['edit_to_temperature_kl']])).all().item() for r in records),prefix_state_replay_max_relative=max(replay),norm_match_max_relative=max(normerrors),group_basis_orthogonality_max_abs=max(ortherrors),zero_dose_max_kl=zero,unique_pair_documents=len(used_docs),directed_pairs=len(pairs),native_endpoints=len(native),whole_model_forwards=forwards,test_evaluations=0)
 assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(normerrors)<1e-5 and zero<1e-7
 torch.save(dict(pairs=pairs,patches=patches,native_logits=native,edited_logits=torch.stack(logitrows)),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,coverage=coverage,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()

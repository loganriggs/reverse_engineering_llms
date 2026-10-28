"""H774 nested data coverage x objective x fixed minibatch schedules."""
import json,time,hashlib
from pathlib import Path
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_l5_refined_output_geometry import forward
from run_bilin18_l5_direct_kl_rank8 import RankEight
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_kl_training_coverage'
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_l5_natural_featurizer_baselines_data.pt','base':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','previous_fits':ART/'bilin18_l5_direct_kl_rank8_fits.pt','previous_result':ART/'bilin18_l5_direct_kl_rank8.json','implementation':Path(__file__).with_name('run_bilin18_l5_direct_kl_rank8.py'),'forward_source':Path(__file__).with_name('run_bilin18_l5_refined_output_geometry.py'),'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);base=torch.load(paths['base'],weights_only=False)['fits']['L5H5'];previous=torch.load(paths['previous_fits'],weights_only=False);old=json.loads(paths['previous_result'].read_text());pool=sorted(range(2048),key=lambda d:hashlib.sha256(f'H773:{d}'.encode()).hexdigest())[:1024];val=old['config']['validation_documents'];assert pool[:128]==old['config']['train_documents'];cfg={'experiment':'H774','hashes':hashes,'training_pool':pool,'pool_sizes':[128,1024],'seeds':[7731,7742,7743],'validation_documents':val,'common_train_evaluation_documents':pool[:128],'lengths':[32,120],'rank':8,'parameters':2048,'steps':128,'batch_size':8,'learning_rate':.001,'normalizers':previous['normalizers'],'design':'Nested128/1024document pools, identical writePCA8 initialization, objectives reconstruction/directKL, three fixed schedules. Same128updates,batch8,LR.001,global normalization for every condition. Schedule paired across objectives within pool/seed. Final checkpoints only; report all conditions. Same64document validation panel and common128document training panel.','limitations':'Reused validation; three schedule seeds not independent datasets; effective exposure limited to1024context draws per run. Objective normalizers fixed to H773 first training batch, no tuning or validation checkpoint selection.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval().requires_grad_(False)
 E=base['write_pca']['encode'].float().cuda();B=base['write_pca']['decode'].float().cuda();mu=base['mean'].float().cuda();scale=previous['scale'];cache={};count=0;backwards=0;captureerr=[]
 with torch.no_grad():
  for n in cfg['lengths']:
   ds=pool+val
   for lo in range(0,len(ds),8):
    bb=ds[lo:lo+8];saved={};out=forward(model,data['all_token_ids'][bb,:n].cuda(),capture=saved).cpu();count+=len(bb)
    for i,d in enumerate(bb):cache[d,n]={'head':saved['head'][i],'logits':out[i]};captureerr.append(float((saved['head'][i]-data['activations']['L5H5'][d,n//8-1]).abs().max()))
   print('cached',n,len(ds),flush=True)
 def batch(ds,n):
  heads=torch.stack([cache[d,n]['head'] for d in ds]).cuda();teacher=torch.stack([cache[d,n]['logits'] for d in ds]).cuda().double().log_softmax(-1);return data['all_token_ids'][ds,:n].cuda(),heads,(heads-mu)@E/scale,teacher
 def kl(out,t):return (t.exp()*(t-out.double().log_softmax(-1))).sum(-1).mean()
 states={'initial':RankEight().state_dict()};metadata={'initial':{'pool_size':0,'seed':0,'objective':'initial'}};logs=[];schedules={};exposure=[]
 for size in cfg['pool_sizes']:
  for seed in cfg['seeds']:
   gen=torch.Generator().manual_seed(seed);schedule=[(cfg['lengths'][i%2],[pool[j] for j in torch.randint(size,(8,),generator=gen).tolist()]) for i in range(128)];schedules[size,seed]=schedule;exposure.append({'pool_size':size,'seed':seed,'unique_documents_seen':len({d for n,ds in schedule for d in ds}),'unique_contexts_seen':len({(d,n) for n,ds in schedule for d in ds})})
   for objective in ['reconstruction','direct_KL']:
    name=f'{objective}_n{size}_s{seed}';net=RankEight().cuda();opt=torch.optim.Adam(net.parameters(),lr=.001)
    for step,(n,ds) in enumerate(schedule):
     tokens,heads,z,teacher=batch(ds,n);opt.zero_grad();rec=net(z)
     if objective=='reconstruction':loss=(rec-z).square().mean()
     else:loss=kl(forward(model,tokens,(rec*scale)@B+mu-heads),teacher);count+=len(ds)
     normalized=loss/previous['normalizers'][objective];assert torch.isfinite(normalized);normalized.backward();backwards+=len(ds) if objective=='direct_KL' else 0;opt.step()
     if step in [0,31,63,95,127]:logs.append({'method':name,'step':step+1,'loss':float(loss.detach())})
    states[name]={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};metadata[name]={'pool_size':size,'seed':seed,'objective':objective};print('trained',name,flush=True);del net,opt
 fitpath=ART/f'{STEM}_fits.pt';torch.save({'states':states,'metadata':metadata,'scale':scale,'config':cfg,'schedules':schedules},fitpath);fithash=sha(fitpath);replay={objective:max(float((states[f'{objective}_n128_s7731'][k]-v).abs().max()) for k,v in previous['states'][objective].items()) for objective in ['reconstruction','direct_KL']};records=[]
 with torch.no_grad():
  for name,state in states.items():
   net=RankEight().cuda();net.load_state_dict(state);net.eval()
   for split,docs in [('common_train',pool[:128]),('validation',val)]:
    for n in cfg['lengths']:
     for lo in range(0,len(docs),8):
      ds=docs[lo:lo+8];tokens,heads,z,teacher=batch(ds,n);rec=net(z);out=forward(model,tokens,(rec*scale)@B+mu-heads).double().log_softmax(-1);count+=len(ds)
      for i,d in enumerate(ds):target=int(data['all_token_ids'][d,n]);records.append({'split':split,'doc':d,'length':n,'method':name,**metadata[name],'KL':float((teacher[i].exp()*(teacher[i]-out[i])).sum()),'CE_change':float(teacher[i,target]-out[i,target]),'write_MSE_scaled':float((rec[i]-z[i]).square().mean())})
   print('evaluated',name,flush=True)
 summary=[]
 for split in ['common_train','validation']:
  for name in states:
   rr=[r for r in records if r['split']==split and r['method']==name];summary.append({'split':split,'method':name,**metadata[name],**{k:sum(r[k] for r in rr)/len(rr) for k in ['KL','CE_change','write_MSE_scaled']}})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'fit_hash_unchanged':sha(fitpath)==fithash,'H773_parameter_replay_max_abs':replay,'cached_head_replay_max_abs':max(captureerr),'model_parameters_frozen':all(not p.requires_grad for mod in [model.embedding,model.blocks,model.lm_head] for p in mod.parameters()),'document_splits_disjoint':not bool(set(pool)&set(val)),'finite':all(torch.isfinite(torch.tensor([r['KL'],r['CE_change'],r['write_MSE_scaled']])).all() for r in records),'whole_model_forwards':count,'KL_backward_context_evaluations':backwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['fit_hash_unchanged'] and audits['model_parameters_frozen'] and audits['document_splits_disjoint'] and audits['finite'] and max(captureerr)<1e-4 and max(replay.values())<1e-5
 (ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'summary':summary,'records':records,'training_log':logs,'exposure':exposure,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits},indent=2),flush=True)
if __name__=='__main__':main()

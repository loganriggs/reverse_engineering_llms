"""H768 crossed context/error directional curvature, without categorical sampling."""
import json,time
from pathlib import Path
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_l5_refined_output_geometry import forward
from run_bilin18_l5_natural_featurizer_baselines import reconstruct
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_cross_context_curvature'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 paths={'data':ART/'bilin18_l5_natural_featurizer_baselines_data.pt','base':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','geometry':ART/'bilin18_l5_refined_output_geometry_captures.pt','native':ART/'bilin18_l5_geometry_output_fidelity_captures.pt','previous':ART/'bilin18_l5_local_global_curvature.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()}
 prev=json.loads(paths['previous'].read_text());docs=list(dict.fromkeys(d for d,n in prev['config']['contexts']));cfg={'experiment':'H768','hashes':hashes,'docs':docs,'lengths':[32,120],'budgets':[8,64],'scales':[-.02,-.01,0.,.01,.02],'design':'Full crossed16x16 source-error and destination-context matrix at each length and budget. Adaptive write-PCA error directions only, no outcome selection. Apply identical source error to every destination, estimate local directional Fisher with full vocabulary variance of numerical derivative. Compare diagonal against mean across destinations, mean excluding same source, and H764 global. Both finite-difference steps retained.','limitations':'Reused small validation panel, length-matched swaps only; no learned conditional metric; numerical derivatives; correlation can arise from error/context coupling or heterogeneous sensitivity.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['data'],weights_only=False);base=torch.load(paths['base'],weights_only=False)['fits']['L5H5'];geo=torch.load(paths['geometry'],weights_only=False);cached=torch.load(paths['native'],weights_only=False);model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 records=[];replay=[];count=0
 for length in cfg['lengths']:
  raw=torch.stack([cached['native_heads'][d,length] for d in docs]).double();errors={k:reconstruct(raw,base,'adaptive_write_pca',k)-raw for k in cfg['budgets']}
  for di,dest in enumerate(docs):
   tokens=data['all_token_ids'][dest,:length][None].expand(5,-1).cuda()
   for k in cfg['budgets']:
    for si,source in enumerate(docs):
     error=errors[k][si];delta=torch.stack([s*error for s in cfg['scales']]).float().cuda();out=forward(model,tokens,delta).cpu().double();count+=5;lp=out[2].log_softmax(-1);p=lp.exp();g1=(out[3]-out[1])/.02;g2=(out[4]-out[0])/.04
     def var(g):return float((p*(g-(p*g).sum()).square()).sum())
     if si==0:replay.append(float((out[2]-cached['native_logits'][dest,length]).abs().max()))
     records.append({'source_doc':source,'destination_doc':dest,'length':length,'budget':k,'curvature':var(g1),'curvature_eps02':var(g2),'global_curvature':float(error@geo['metric']@error)})
   print('length',length,'destination',di+1,'forwards',count,flush=True)
 summary=[];matrices={};diag_replay=[]
 for length in cfg['lengths']:
  for k in cfg['budgets']:
   rr=[r for r in records if r['length']==length and r['budget']==k];C=torch.tensor([[next(r['curvature'] for r in rr if r['source_doc']==s and r['destination_doc']==d) for d in docs] for s in docs],dtype=torch.double);D=torch.tensor([[next(r['curvature_eps02'] for r in rr if r['source_doc']==s and r['destination_doc']==d) for d in docs] for s in docs],dtype=torch.double);actual=C.diag();mean=C.mean(1);loo=(C.sum(1)-actual)/(len(docs)-1);global_pred=torch.tensor([next(r['global_curvature'] for r in rr if r['source_doc']==s) for s in docs],dtype=torch.double)
   old=torch.tensor([next(r['local_curvature'] for r in prev['directional_curvature'] if r['doc']==d and r['length']==length and r['budget']==k and r['variant']=='adaptive_write_pca') for d in docs],dtype=torch.double);diag_replay.append(float((actual-old).norm()/old.norm()))
   row={'length':length,'budget':k,'matched_mean':float(actual.mean()),'off_diagonal_mean':float(loo.mean()),'matched_to_off_diagonal_ratio':float(actual.mean()/loo.mean()),'finite_difference_relative_RMS':float((C-D).norm()/D.norm())}
   for name,pred in [('same_panel_mean',mean),('other_context_mean',loo),('old_global',global_pred)]:row[name+'_relative_RMS_error']=float((pred-actual).norm()/actual.norm());row[name+'_median_relative_error']=float(((pred-actual).abs()/actual).median())
   # Remove global destination scale to assess direction-specific context interaction.
   normalized=C/C.mean(0,keepdim=True);row['destination_normalized_matched_to_off_diagonal_ratio']=float(normalized.diag().mean()/((normalized.sum()-normalized.diag().sum())/(len(docs)*(len(docs)-1))))
   summary.append(row);matrices[length,k]={'curvature':C,'curvature_eps02':D}
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_replay_max_abs':max(replay),'diagonal_H767_relative_RMS_max':max(diag_replay),'finite':all(torch.isfinite(m['curvature']).all().item() for m in matrices.values()),'whole_model_forwards':count,'test_evaluations':0}
 assert audits['hashes_unchanged'] and audits['finite'] and max(replay)<1e-4 and max(diag_replay)<.02
 torch.save({'matrices':matrices,'docs':docs},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits},indent=2),flush=True)
if __name__=='__main__':main()

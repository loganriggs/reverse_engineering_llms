"""Posthoc training-only error concentration; no fitting or sample removal."""
import json
import numpy as np
import torch
from bilinear_input_runtime import InputWritePredictor
from fit_pca_encoder import sha,V,A

def main():
 torch.set_num_threads(1);ref=json.loads((V/'native_input_reference.json').read_text());s=torch.load(ref['path'],weights_only=True);m4=json.loads((V/'natural_token_context.json').read_text());data=torch.load(m4['cache'],weights_only=True);ix=s['indices'];z=s['z'][ix];w=data['Y'][1][ix];initials=json.loads((V/'bilinear_input_initial_states.json').read_text())['models'];results=json.loads((V/'bilinear_input_fullbatch_results.json').read_text());rows=[]
 for e in results['models']:
  assert sha(e['path'])==e['sha256'];state=torch.load(e['path'],weights_only=True);start=next(v for v in initials if v['kind']==e['kind'] and v['seed']==e['seed']);init=torch.load(start['path'],weights_only=True);f=InputWritePredictor(init);initerrs=[];finalerrs=[]
  with torch.no_grad():
   for dest,weights in [(initerrs,None),(finalerrs,state['state'])]:
    if weights is not None:f.load_state_dict(weights)
    for offset in range(0,len(z),128):
     pred,_,_=f(z[offset:offset+128]);dest.extend(((pred-w[offset:offset+128])/f.output_scale).double().square().mean(-1).tolist())
  er=np.array(finalerrs);order=np.argsort(-er);x=(z-f.input_mean)/f.input_scale;norm=x.double().norm(dim=-1).numpy();top=[]
  for j in order[:10]:top.append(dict(training_index=int(j),cache_index=int(ix[j]),document=int(data['documents'][ix[j]]),position=int(ix[j])%128,token_id=int(data['tokens'][ix[j]]),normalized_input_norm=float(norm[j]),initial_error=initerrs[j],final_error=finalerrs[j]))
  row=dict(kind=e['kind'],seed=e['seed'],initial_full_MSE=float(np.mean(initerrs)),final_full_MSE=float(er.mean()),top1_error_share=float(er[order[:1]].sum()/er.sum()),top10_error_share=float(er[order[:10]].sum()/er.sum()),median_position_error=float(np.median(er)),top_positions=top);rows.append(row);print(e['kind'],e['seed'],row['initial_full_MSE'],row['final_full_MSE'],row['top1_error_share'],top[0],flush=True)
 out=dict(status='completed',source_sha256=sha(__file__),warmup_results_sha256=sha(V/'bilinear_input_fullbatch_results.json'),models=rows,scope='Posthoc descriptive fulltraining error concentration and input norms; allpositions retained. No newfit, hyperparameter selection or heldout data. Extremes do not establish their cause without numerical and optimizer diagnostics.')
 p=V/'bilinear_input_fullbatch_error_diagnostic.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()

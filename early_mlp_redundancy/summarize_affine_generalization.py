"""M78 like-for-like pooled raw error alongside document-average error."""
import json
import torch
from fit_pca_encoder import sha,V

def main():
    torch.set_num_threads(1);p=V/'affine_input_natural.json';r=json.loads(p.read_text());a=json.loads((V/'affine_input_natural_audit.json').read_text());assert a['status']=='passed' and a['result_sha256']==sha(p)
    error=0.;energy=0.;rows=[]
    for file in r['files']:
        assert sha(file['path'])==file['sha256'];s=torch.load(file['path'],weights_only=True);assert len(s['conditions'])==1;w=s['write'].double();q=s['conditions'][0]['reconstruction'].double();er=float((w-q).square().sum());en=float(w.square().sum());error+=er;energy+=en;rows.append(dict(document=s['document'],error_energy=er,target_energy=en,relative_error=er/en))
    out=dict(status='completed',source_sha256=sha(__file__),natural_sha256=sha(p),training_fit_sha256=sha(V/'affine_input_fit.json'),training_pooled_relative_error=json.loads((V/'affine_input_fit.json').read_text())['train_relative_raw_error'],heldout_pooled_relative_error=error/energy,heldout_mean_document_relative_error=sum(x['relative_error'] for x in rows)/len(rows),rows=rows)
    (V/'affine_input_generalization.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='rows'})
if __name__=='__main__':main()

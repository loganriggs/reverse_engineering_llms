"""Training-document check of all ten evaluation adapters against independent algebra."""
import json
import torch
from fit_pca_encoder import V,sha
from product_dictionary_evaluation import ProductDictionaryPredictor
from independent_dictionary_predictions import IndependentDictionaryPrediction

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');mp=V/'product_dictionary_coverage_evaluation_models.json';m=json.loads(mp.read_text())
 for name,digest in m['references'].items():assert sha(V/name)==digest
 assert m['runtime_sha256']==sha(V.parent/'product_dictionary_featurizer.py')
 ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];r=torch.load(ref['path'],weights_only=True);z=r['z'][:128][None].cuda();rows=[]
 for entry in m['models']:
  enc=ProductDictionaryPredictor(entry);ind=IndependentDictionaryPrediction(entry);q,s,c,info=enc.reconstruct(z,object());de,ce=ind.verify(ind.prepare(z),q,s,c);assert not q.requires_grad and s.shape==c.shape==(1,128,128);assert info['output_dimension']==128 and info['retained_native_products']==4608
  rows.append({k:entry[k] for k in ['kind','seed','arm']}|dict(prediction_relative_error=de,coordinate_relative_error=ce));print('CHECKED',entry['seed'],entry['arm'],flush=True)
 out=dict(status='passed',source_sha256=sha(__file__),models_sha256=sha(mp),adapter_sha256=sha(V.parent/'product_dictionary_evaluation.py'),independent_helper_sha256=sha(V.parent/'independent_dictionary_predictions.py'),rows=rows,scope='All10adapters on trainingdoc0 only, unusable native-write argument ignored; independent float64 NumPy prediction/coordinate replay. Native forward guards were checked during M98preflight; no new nativeforwards.')
 path=V/'product_dictionary_coverage_evaluation_preflight.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()

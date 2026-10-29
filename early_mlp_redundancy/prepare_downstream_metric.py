"""Freeze M74 derivative preflight and evaluation inventory before full scoring."""
import json
from pathlib import Path
from fit_pca_encoder import sha
P=Path(__file__).resolve().parent;V=P/'validation'

def main():
    protocol=json.loads((V/'downstream_residual_metric_protocol.json').read_text())
    for name,digest in protocol['references'].items():assert sha(V/name)==digest
    original=json.loads((V/'least_squares_evaluation_models.json').read_text())['models']
    models=[dict(kind='pca',seed=0,arm='rank128',variant='pca',rank=128)]
    models += [e for e in original if e['variant']=='full_dictionary_LS']
    assert len(models)==19
    bindings={name:sha(P/name) for name in ['downstream_residual_metric.py','pca_encoder.py','least_squares_encoder.py','train_write_featurizers.py','inspect_model.py']}
    result=dict(protocol_sha256=sha(V/'downstream_residual_metric_protocol.json'),source_hashes=bindings,
                models=models,preflight_documents=[32,63],finite_difference_steps=[.01,.003,.001],
                preflight_max_relative_derivative_error=.03,preflight_max_relative_Fisher_error=.01,
                evaluation_documents=list(range(32,64)),amplitudes=[.25,.5,1.],
                prior_KL_replay_atol=1e-5,prior_KL_replay_rtol=1e-4,
                arithmetic='Float32 residual reconstruction-native, patched as native+alpha*residual. This may differ by rounding from prior reconstruction+bias replacement. Verify prior KL with declared tolerance.',
                smoke='PCA128 doc32 forwardAD succeeded with exactly equal primal; finite-difference relative errors .000786/.002534/.007541 at .01/.003/.001. No outcome-based step changes.',
                full_output_policy='Save native outputs per document and all19 candidate tangents and logits at each of three amplitudes; independently rescore all outputs.')
    path=V/'downstream_metric_execution.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print('Frozen19encoders,38preflightcases,608evaluationcases')
if __name__=='__main__':main()

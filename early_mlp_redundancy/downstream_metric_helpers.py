"""Hash-bound M74 data and encoder construction."""
import json
from pathlib import Path
import torch
from fit_pca_encoder import sha
from pca_encoder import PCAEncoder
from least_squares_encoder import LeastSquaresEncoder
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')

def setup():
    execution=json.loads((V/'downstream_metric_execution.json').read_text())
    assert execution['protocol_sha256']==sha(V/'downstream_residual_metric_protocol.json')
    protocol=json.loads((V/'downstream_residual_metric_protocol.json').read_text())
    for name,digest in protocol['references'].items():assert sha(V/name)==digest
    for name,digest in execution['source_hashes'].items():assert sha(P/name)==digest
    encoders=[]
    for e in execution['models']:
        if e['kind']=='pca':enc=PCAEncoder(128)
        else:
            assert sha(e['path'])==e['sha256'];state=torch.load(e['path'],weights_only=True)
            f=Featurizer(e['kind'],e['seed']).cuda();f.load_state_dict(state['state']);f.requires_grad_(False)
            enc=LeastSquaresEncoder(f,state['mean'].cuda(),state['scale'].cuda())
        encoders.append((e,enc))
    cohort=json.loads((V/'natural_token_context_cohort.json').read_text())
    cache=json.loads((V/'natural_token_context.json').read_text());assert sha(cache['cache'])==cache['cache_sha256']
    writes=torch.load(cache['cache'],weights_only=True)['Y'][1]
    return execution,encoders,cohort,writes


def identity(e):return {k:e[k] for k in ['kind','seed','arm']}

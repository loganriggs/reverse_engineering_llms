"""Held-out QK2-constant branch edit for shared-key subspaces."""

import json

import numpy as np
import torch

from flagship_function_aware_shared_key import make_panels, error
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


def run():
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    panels,gate,coverage=make_panels(state,128,128,8192,
                                     test_seed_offset=10000)
    del state
    prior=np.load(ART/'all_head_shared_key_dictionary_bases.npz')
    methods={
        'shared_uniform_256':prior['shared_uniform'][:,:256],
        'shared_usage_256':prior['shared_half_corpus'][:,:256],
        'shared_usage_384':prior['shared_half_corpus'][:,:384],
    }
    for label,stem in [
        ('function_aware_small_256','all_head_function_aware_shared_key_s641_n8192_t160'),
        ('function_aware_large_256','all_head_function_aware_shared_key_s642_n32768_t160'),
        ('intervention_aware_half_256','all_head_function_aware_shared_key_s644_n32768_t160_branch50'),
        ('intervention_aware_only_256','all_head_function_aware_shared_key_s645_n32768_t160_branch100'),
    ]:
        methods[label]=np.load(ART/f'{stem}.npz')['basis']
    methods={name:torch.from_numpy(value.copy())
             for name,value in methods.items()}
    result={}
    with torch.no_grad():
        for law,panel in panels.items():
            if not law.startswith('test_'):continue
            normal={name:float(error(panel,basis))
                    for name,basis in methods.items()}
            constant=panel.copy()
            constant['s2']=torch.ones_like(panel['s2'])
            constant['ref']=torch.einsum('nh,nhj,nj->n',
                          panel['s1'],panel['gram'],panel['s1']).clamp_min(0)
            edited={name:float(error(constant,basis))
                    for name,basis in methods.items()}
            result[law]={'natural_qk2':normal,'qk2_constant_one':edited,
                         'n_pairs':len(panel['k'])}
            print(law,'natural',normal,'qk2_constant',edited,flush=True)
    output=dict(scope='all-head pair-output response, not full-model propagation',
                intervention='replace both QK2 score factors with 1 for every tested pair; preserve QK1, OV and head sum',
                test_seed_offset=10000,formula_gate=gate,coverage=coverage,
                results=result)
    path=ART/'all_head_function_aware_shared_key_qk2_constant_n8192.json'
    path.write_text(json.dumps(output,indent=2)+'\n')
    print('saved',path)


if __name__=='__main__':run()

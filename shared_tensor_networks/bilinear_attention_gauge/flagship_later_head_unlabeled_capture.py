"""Unlabeled natural head activations in the same fixed rank-4 subspaces."""

import json

import numpy as np
import torch

from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_rank_pilot import HeadCut, contexts, final_logits
from flagship_full_model_eval import GPT, GPTConfig
from flagship_matrix_free import CORPUS, SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART

SEED=20261102
TRAIN=512
TEST=128
LENGTHS=(5,32)


def run():
    torch.set_num_threads(4)
    rng=np.random.default_rng(SEED)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probes={layer:HeadCut(model.transformer.h[layer].attn)
            for layer in sorted(set(layer for layer,_ in TARGETS.values()))}
    bases=np.load(ART/'later_head_rank_pilot_bases.npz')
    arrays={}
    with torch.no_grad():
        for length in LENGTHS:
            panels={'fineweb_train':contexts(CORPUS,rng,TRAIN,range(0,16000),length),
                    'fineweb_test':contexts(CORPUS,rng,TEST,range(17500,20000),length)}
            for name,path in DATASETS.items():
                panels[name]=contexts(path,rng,TEST,range(0,1200),length)
            for panel,ids in panels.items():
                collected={target:[] for target in TARGETS}
                for batch in ids.split(16):
                    final_logits(model,batch,stop_at=max(probes))
                    for target,(layer,head) in TARGETS.items():
                        p=probes[layer];a=head*p.head_dim
                        b=torch.tensor(bases[f'{target}__weight'][:,:4].copy(),dtype=torch.float32)
                        z=(p.last_input[:,-1,a:a+p.head_dim]@b).numpy()
                        collected[target].append(z)
                for target in TARGETS:
                    arrays[f'{target}__{panel}__l{length}']=np.concatenate(collected[target])
                arrays[f'ids__{panel}__l{length}']=ids.numpy()
                print(length,panel,len(ids),flush=True)
    np.savez(ART/'later_head_unlabeled_activations.npz',**arrays)
    (ART/'later_head_unlabeled_activations.json').write_text(json.dumps(dict(
        description='Unlabeled last-position activations in fixed weight-rank-4 later-head subspaces',
        seed=SEED,lengths=LENGTHS,train_contexts=TRAIN,test_contexts=TEST,
        fineweb_train_docs=[0,16000],fineweb_test_docs=[17500,20000],
        corpora=['fineweb','wikitext103','tinystories'],
        arrays={k:list(v.shape) for k,v in arrays.items()}),indent=2)+'\n')


if __name__=='__main__':run()

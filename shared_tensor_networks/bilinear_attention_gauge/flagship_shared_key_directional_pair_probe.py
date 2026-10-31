"""Exact selected-pair contribution for the four-condition directional test."""

import json

import numpy as np
import torch

from flagship_matrix_free import HD,SNAPSHOT,rotary
from flagship_shared_key_dictionary import HEADS,project_all_heads,rotate_all
from flagship_shared_key_counterfactual_causality import norm2
from flagship_transfer import ART


SOURCE='all_head_shared_key_directional_reversal_n8_s111-117.json'


def run():
    panel=json.loads((ART/SOURCE).read_text())
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    basis=np.load(ART/'all_head_function_aware_shared_key_s642_n32768_t160.npz')['basis'].astype(np.float64)
    rotations=np.load(ART/'all_head_shared_key_gauge_priors_function_n2048_p2048_i120_d600.npz')
    qids=np.unique([t for row in panel['fidelity']
                    for t in (row['original_query_id'],row['query_punctuation_id'])])
    kids=np.unique([t for row in panel['fidelity']
                    for t in (row['original_key_id'],row['key_punctuation_id'])])
    query=project_all_heads(state,qids,('c_q','c_q2'))
    key=project_all_heads(state,kids,('c_k','c_k2','c_v'))
    out=state['transformer.h.0.attn.c_proj.weight'].to(torch.float64).numpy()
    cos,sin=rotary(128)
    axes={}
    for name,gauge,component in (
        ('forward_c16','response_0',16),
        ('reverse_c165','response_0',165),
        ('random_c94','random_0',94),
        ('random_c106','random_0',106)):
        axes[name]=(basis@rotations[gauge])[:,component].reshape(HEADS,HD)
    records=[]
    for row in panel['fidelity']:
        i=row['query_position'];j=row['key_position']
        outputs={}
        for condition,query_id,key_id in (
            ('word_query_word_key',row['original_query_id'],row['original_key_id']),
            ('word_query_punct_key',row['original_query_id'],row['key_punctuation_id']),
            ('punct_query_word_key',row['query_punctuation_id'],row['original_key_id']),
            ('punct_query_punct_key',row['query_punctuation_id'],row['key_punctuation_id'])):
            qi=np.searchsorted(qids,query_id);ki=np.searchsorted(kids,key_id)
            qpost=rotate_all(query['c_q'][qi][None],np.array([i]),cos,sin)[0]
            qeff=rotate_all(qpost[None],np.array([j]),cos,-sin)[0]
            q2post=rotate_all(query['c_q2'][qi][None],np.array([i]),cos,sin)[0]
            k2post=rotate_all(key['c_k2'][ki][None],np.array([j]),cos,sin)[0]
            s2=np.sum(q2post*k2post,axis=-1)/HD
            kpre=key['c_k'][ki]
            value=key['c_v'][ki]
            ov=np.stack([value[h]@out[:,h*HD:(h+1)*HD].T
                         for h in range(HEADS)],axis=0)
            for name,axis in axes.items():
                coord=float(np.sum(kpre*axis))
                score=np.sum(qeff*axis,axis=-1)*coord*s2/HD
                outputs[(name,condition)]=score@ov
        for name in axes:
            word_change=(outputs[(name,'word_query_punct_key')]-
                         outputs[(name,'word_query_word_key')])
            punct_change=(outputs[(name,'punct_query_punct_key')]-
                          outputs[(name,'punct_query_word_key')])
            dose=panel['detail'][name]['dose']
            records.append(dict(corpus=row['corpus'],seed=row['seed'],
                document_id=row['document_id'],method=name,
                dose=dose,
                word_query_key_edit_pair_energy=norm2(word_change)*dose*dose,
                punct_query_key_edit_pair_energy=norm2(punct_change)*dose*dose,
                word_query_punct_key_pair_energy=norm2(
                    outputs[(name,'word_query_punct_key')])*dose*dose,
                punct_query_word_key_pair_energy=norm2(
                    outputs[(name,'punct_query_word_key')])*dose*dose))
    result=dict(source=SOURCE,layer=0,rank=256,
        scope='one selected key-query pair contribution under QK1, QK2, OV, RoPE and nine-head sum; no other context keys or downstream layers',
        records=records)
    (ART/'all_head_shared_key_directional_pair_probe_n48.json').write_text(
        json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()

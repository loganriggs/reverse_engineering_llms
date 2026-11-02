"""Exact data-free token-pair function for one first-layer bilinear head.

At fixed query/key positions i>=j, the head contribution is
S_h(token_i,token_j;i,j) * P_h(token_j), where S_h is the product of
two RoPE-aware QK scores and P_h is the OV value vector. No sampled text is
used to build the tables. A masked-V forward pass independently checks the
identity on randomly chosen token pairs.
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def rotated_all(attn, values, position, length, apply_rotary_emb):
    v, h, d = values.shape
    cos, sin = attn.rotary(torch.empty((1, length, h, d), device=values.device))
    return apply_rotary_emb(values[:, None], cos[:, position:position+1],
                            sin[:, position:position+1])[:, 0]


@torch.no_grad()
def token_function(model, query_position, key_position, apply_rotary_emb):
    assert 0 <= key_position <= query_position < model.config.seq_len
    attn = model.blocks[0].attn
    x = model.blocks[0].norm_attn(model.embed.weight)
    vocab = len(x)
    heads, head_dim = attn.n_head, attn.head_dim
    shape = (vocab, heads, head_dim)
    length = query_position + 1
    q1 = rotated_all(attn, attn.q1(x).reshape(shape), query_position,
                     length, apply_rotary_emb)
    k1 = rotated_all(attn, attn.k1(x).reshape(shape), key_position,
                     length, apply_rotary_emb)
    q2 = rotated_all(attn, attn.q2(x).reshape(shape), query_position,
                     length, apply_rotary_emb)
    k2 = rotated_all(attn, attn.k2(x).reshape(shape), key_position,
                     length, apply_rotary_emb)
    value = attn.v(x).reshape(shape)
    if attn.qk_norm:
        q1, k1, q2, k2 = (torch.nn.functional.normalize(z, dim=-1)
                          for z in (q1, k1, q2, k2))
    first = torch.einsum('ahd,bhd->hab', q1, k1) / head_dim
    second = torch.einsum('ahd,bhd->hab', q2, k2) / head_dim
    score = first * second
    ov = torch.stack([value[:, h] @ attn.out.weight[:, h*head_dim:(h+1)*head_dim].T
                      for h in range(heads)])
    return dict(q1=q1, k1=k1, q2=q2, k2=k2,
                first=first, second=second, score=score, ov=ov)


@torch.no_grad()
def verify(model, tables, query_position, key_position, n, seed):
    attn = model.blocks[0].attn
    gen = torch.Generator().manual_seed(seed)
    vocab = model.config.vocab_size
    length = query_position + 1
    ids = torch.randint(vocab, (n, length), generator=gen)
    # Fix the tested query/key IDs independently; all other tokens are random.
    query = ids[:, query_position]
    key = ids[:, key_position]
    x = model.blocks[0].norm_attn(model.embed(ids))

    def keep_only_key(_module, _inputs, values):
        masked = torch.zeros_like(values)
        masked[:, key_position] = values[:, key_position]
        return masked

    hook = attn.v.register_forward_hook(keep_only_key)
    try:
        actual = attn(x)[:, query_position]
    finally:
        hook.remove()
    predicted = (tables['score'][:, query, key, None] *
                 tables['ov'][:, key]).sum(0)
    absolute = (actual-predicted).abs()
    return dict(n=n, max_abs_error=float(absolute.max()),
                mean_l2_error=float(absolute.square().sum(-1).sqrt().mean()),
                max_relative_l2_error=float(
                    ((actual-predicted).square().sum(-1).sqrt()/
                     actual.square().sum(-1).sqrt().clamp_min(1e-8)).max()))


def ranks(matrix, target_fracs=(.9,.99,.999)):
    singular = torch.linalg.svdvals(matrix.double())
    energy = singular.square()
    total = energy.sum()
    cumulative = energy.cumsum(0)/total.clamp_min(1e-30)
    required = {str(frac): int(torch.searchsorted(cumulative, frac).item()+1)
                for frac in target_fracs}
    errors = {}
    for rank in (1,2,4,8,16,32,64,128,256):
        if rank <= len(singular):
            errors[str(rank)] = float((1-cumulative[rank-1]).clamp_min(0).sqrt())
    return dict(rank_at_energy=required, relative_frobenius_tail=errors,
                top_singular_values=[float(z) for z in singular[:12]])


def gauge_check(tables, head_dim, seed):
    # An invertible QK-bond basis change leaves each score matrix fixed.
    gen = torch.Generator().manual_seed(seed)
    q, k = tables['q1'][0], tables['k1'][0]
    gauge = torch.randn((head_dim,head_dim),generator=gen,dtype=torch.float64)
    gauge += 2*torch.eye(head_dim,dtype=torch.float64)
    qnew = q.double() @ gauge
    knew = k.double() @ torch.linalg.inv(gauge).T
    original = q.double() @ k.double().T / head_dim
    changed = qnew @ knew.T / head_dim
    return float((original-changed).abs().max())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir',default='language_task_suite/results/v8_scope_balanced_continue_seed101')
    p.add_argument('--query-position',type=int,default=31)
    p.add_argument('--key-position',type=int,default=7)
    p.add_argument('--verify-pairs',type=int,default=32)
    p.add_argument('--translation-shift',type=int,default=32,
                   help='Compare an equal-offset pair shifted by this many positions')
    p.add_argument('--seed',type=int,default=52001)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config,apply_rotary_emb
    folder=Path(a.model_dir)
    saved=json.loads((folder/'results.json').read_text())
    record=saved['records'][0]
    model=BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(folder/f"model_depth{record['depth']}.pt",
                                     map_location='cpu',weights_only=True))
    model.eval()
    assert model.use_rmsnorm and not model.config.use_qk_norm
    tables=token_function(model,a.query_position,a.key_position,apply_rotary_emb)
    check=verify(model,tables,a.query_position,a.key_position,a.verify_pairs,a.seed)
    if check['max_abs_error']>1e-5:
        raise AssertionError(f"Token-function replay failed: {check}")
    translated=None
    if a.translation_shift:
        q2=a.query_position+a.translation_shift
        k2=a.key_position+a.translation_shift
        if q2>=model.config.seq_len or k2<0:
            p.error('Translated positions are outside the model sequence length')
        shifted=token_function(model,q2,k2,apply_rotary_emb)
        translated=dict(query_position=q2,key_position=k2)
        for name in ('first','second','score'):
            delta=shifted[name]-tables[name]
            translated[name+'_max_abs']=float(delta.abs().max())
            translated[name+'_relative_frobenius']=float(
                delta.norm()/tables[name].norm().clamp_min(1e-30))
    heads=[]
    for h in range(model.config.n_head):
        score=tables['score'][h]
        # Query unfolding of T[a,b,o]=score[a,b]*OV[b,o].
        # Its Gram matrix equals (score*||OV_b||)(score*||OV_b||)^T.
        weighted=score*tables['ov'][h].norm(dim=-1)[None,:]
        heads.append(dict(head=h,first_qk=ranks(tables['first'][h]),
                          second_qk=ranks(tables['second'][h]),
                          collapsed_score=ranks(score),
                          output_query_unfolding=ranks(weighted),
                          score_frobenius_norm=float(score.norm()),
                          output_query_unfolding_norm=float(weighted.norm())))
    result=dict(config=vars(a),verification=check,equal_offset_shift=translated,
                qk_bond_gauge_max_error=gauge_check(tables,model.blocks[0].attn.head_dim,a.seed+1),
                heads=heads,
                caveat='Exact only for layer 0 at the specified absolute query/key '
                       'positions. RoPE cached trigonometry is bfloat16, so even '
                       'equal offsets need not be numerically identical at all '
                       'positions. The effective score and OV map are functional '
                       'objects; their SVD is not a unique semantic decomposition '
                       'or a causal account of the full three-layer model.')
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    table_path=path.with_suffix('.pt')
    torch.save({name:tables[name] for name in ('first','second','score','ov')},
               table_path)
    result['table_checkpoint']=str(table_path)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('replay',check,'gauge',result['qk_bond_gauge_max_error'],flush=True)
    for head in heads:
        print(head['head'],head['collapsed_score']['rank_at_energy'],
              head['output_query_unfolding']['rank_at_energy'],flush=True)


if __name__=='__main__':main()

"""First-layer bilinear attention: exact token table and loop-aware bond truncation.

The targeted bond joins the two factors of QK1 *after* RoPE. QK2 and OV
remain fixed. A repeated token index in QK1, QK2 and OV is a copy tensor;
thus the bond is not a bridge in the token-function network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import numpy as np
import scipy.linalg as sla


MODEL_ROOT = Path("/home/loganriggs/Coding/reverse_engineer/tensor_language")
DEFAULT_CHECKPOINT = MODEL_ROOT / "runs/L1_d16_h1/model.pt"


def load_model(checkpoint: Path, length: int):
    sys.path.insert(0, str(MODEL_ROOT))
    from model import CycleModel

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    vocab, width = state["embed.weight"].shape
    history_path = checkpoint.with_name("history.json")
    heads = 1
    if history_path.exists():
        cfg = json.loads(history_path.read_text())["config"]
        assert cfg["n_layer"] == 1, cfg
        heads = cfg["n_head"]
    model = CycleModel(vocab, width, heads, 1, length).to(dtype=torch.float64)
    model.load_state_dict(state)
    model.eval()
    return model


def factors(model, length: int, head: int = 0, swap: bool = False):
    a = model.layers[0]
    e = model.embed.weight.detach()
    with torch.no_grad():
        sl = slice(head*a.d_head,(head+1)*a.d_head)
        q1 = (e @ a.q1.weight.T)[:,sl]
        k1 = (e @ a.k1.weight.T)[:,sl]
        q2 = (e @ a.q2.weight.T)[:,sl]
        k2 = (e @ a.k2.weight.T)[:,sl]
        c = ((e @ a.v.weight.T)[:,sl]) @ a.o.weight[:,sl].T
        # RoPE is applied independently to each occurrence. Fix key position 0;
        # RoPE's group property then makes every score depend on delta=i-j.
        def rotate_at(z, pos):
            cos = a.rotary.cos_cached[0, pos, 0]
            sin = a.rotary.sin_cached[0, pos, 0]
            left, right = z.chunk(2, dim=-1)
            return z * cos + torch.cat([-right, left], dim=-1) * sin

        u1 = torch.stack([rotate_at(q1, pos) for pos in range(length)])
        v1 = torch.stack([rotate_at(k1, pos) for pos in range(length)])
        u2 = torch.stack([rotate_at(q2, pos) for pos in range(length)])
        v2 = torch.stack([rotate_at(k2, pos) for pos in range(length)])
        u,v,other_u,other_v = (u2,v2,u1,v1) if swap else (u1,v1,u2,v2)
        b = torch.einsum("iar,jbr->ijab", other_u, other_v) / (a.d_head**2)
        b *= torch.tril(torch.ones(length, length, dtype=b.dtype))[:, :, None, None]
    return [x.numpy() for x in (u, v, b, c)]


def token_table(u, v, b, c, x):
    # x is the matrix inserted on the QK1 bond. In the original model x=I.
    score = np.einsum("iar,rs,jbs->ijab", u, x, v, optimize=True)
    return np.einsum("ijab,ijab,bo->ijabo", score, b, c, optimize=True)


def pair_measure(length, token_probs):
    vocab = len(token_probs)
    pair = np.zeros((length,length,vocab,vocab))
    independent = np.outer(token_probs,token_probs)
    diagonal = np.diag(token_probs)
    for i in range(length):
        for j in range(i+1):
            pair[i,j] = diagonal if i == j else independent
    pair /= length*(length+1)/2
    return pair


def gram_environment(u, v, b, c, measure=None):
    # Gram matrix of the linear map X -> token_table(X); no full fifth-order
    # coefficient tensor and no V^2-by-d^2 design matrix are materialized.
    weights = b**2 * np.sum(c**2, axis=1)[None, None, None, :]
    if measure is not None:
        weights *= measure
    env4 = np.einsum(
        "iar,iat,jbs,jbu,ijab->rstu", u, u, v, v, weights, optimize=True
    )
    d = v.shape[-1]
    return env4.reshape(d * d, d * d)


def local_svd(u, v, rank):
    """Cycle-cut baseline: SVD of QK1 with other paths disconnected."""
    d = v.shape[-1]
    left = u.reshape(-1, d)
    right = v.reshape(-1, d)
    qk = left @ right.T
    uu, ss, vh = sla.svd(qk, full_matrices=False)
    approx = (uu[:, :rank] * ss[:rank]) @ vh[:rank]
    return sla.pinv(left) @ approx @ sla.pinv(right.T)


def dominant_positive_operator(t):
    vals, vectors = sla.eig(t)
    idx = np.argmax(np.abs(vals))
    lam = vals[idx]
    op = vectors[:, idx].real.reshape(int(np.sqrt(t.shape[0])), -1)
    op = 0.5 * (op + op.T)
    if np.trace(op) < 0:
        op = -op
    op /= np.trace(op)
    return lam, op, vals


def weighted_trace_gauge(env, rank):
    """Evenbly 2018 Eqs. 3-8 for a real bond with sigma=I."""
    d = int(np.sqrt(env.shape[0]))
    transfer = env.reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(d*d, d*d)
    lam_l, left, eigenvalues = dominant_positive_operator(transfer)
    lam_r, right, _ = dominant_positive_operator(transfer.T)
    assert abs(lam_r - lam_l) < 1e-7 * max(1, abs(lam_r))
    eig_l, ul = sla.eigh(left)
    eig_r, ur = sla.eigh(right)
    if min(eig_l.min(), eig_r.min()) <= 1e-12:
        raise ValueError("WTG does not exist numerically: non-positive boundary operator")
    sqrt_l = (ul * np.sqrt(eig_l)) @ ul.T
    sqrt_r = (ur * np.sqrt(eig_r)) @ ur.T
    wl, singular, wrt = sla.svd(sqrt_l @ sqrt_r)
    x = wl.T @ sqrt_l
    y = sqrt_r @ wrt.T
    sigma_cut = np.diag(np.r_[singular[:rank], np.zeros(d-rank)])
    bond_cut = sla.solve(x, sigma_cut) @ sla.inv(y)
    probs = np.abs(eigenvalues)
    probs /= probs.sum()
    entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
    return bond_cut, singular, float(entropy), x, y


def cut_wtg(x,y,singular,rank):
    d = len(singular)
    sigma_cut = np.diag(np.r_[singular[:rank],np.zeros(d-rank)])
    return sla.solve(x,sigma_cut) @ sla.inv(y)


def wtg_boundary_check(u,v,b,c,measure,x,y,singular):
    invx = sla.inv(x)
    invy = sla.inv(y)
    unew = u @ invx
    vnew = v @ invy.T
    sigma = np.diag(singular)
    original = token_table(u,v,b,c,np.eye(v.shape[-1]))
    regauged = token_table(unew,vnew,b,c,sigma)
    reconstruction = float(np.max(np.abs(original-regauged)))
    e4 = gram_environment(unew,vnew,b,c,measure).reshape((len(singular),)*4)
    rho_l = np.einsum("rstu,ks,ku->rt",e4,sigma,sigma,optimize=True)
    rho_r = np.einsum("rstu,rk,tk->su",e4,sigma,sigma,optimize=True)
    n = len(singular)
    eye = np.eye(n)/n
    defect = max(np.max(np.abs(rho_l/np.trace(rho_l)-eye)),
                 np.max(np.abs(rho_r/np.trace(rho_r)-eye)))
    return reconstruction,float(defect)


def env_error(env, x):
    d = x.shape[0]
    diff = (x - np.eye(d)).reshape(-1)
    target = np.eye(d).reshape(-1)
    return float((diff @ env @ diff) / (target @ env @ target))


def literal_error(table, approximation, measure):
    return float(np.sum(measure[...,None]*(approximation-table)**2)
                 /np.sum(measure[...,None]*table**2))


def fet_als(env, initial, rank, iterations=200):
    """Full-environment alternating rank-r least squares (FET objective).

    Unlike the paper's fidelity-normalized update, squared-error minimization
    optimizes the same direction and its best scale. All contractions use only
    the bond environment. No claim of globally optimal rank-r solution.
    """
    d = initial.shape[0]
    u, s, vh = sla.svd(initial, full_matrices=False)
    left = u[:, :rank] * np.sqrt(s[:rank])
    right = vh[:rank].T * np.sqrt(s[:rank])
    target = env @ np.eye(d).reshape(-1)
    trace = []
    for _ in range(iterations):
        # vec(left @ right.T), C-order, as a linear map of right or left.
        ar = np.zeros((d*d, d*rank))
        for j in range(d):
            for k in range(rank):
                ar[:, j*rank+k] = np.outer(left[:, k], np.eye(d)[j]).reshape(-1)
        right = sla.lstsq(ar.T @ env @ ar, ar.T @ target)[0].reshape(d, rank)
        al = np.zeros((d*d, d*rank))
        for i in range(d):
            for k in range(rank):
                al[:, i*rank+k] = np.outer(np.eye(d)[i], right[:, k]).reshape(-1)
        left = sla.lstsq(al.T @ env @ al, al.T @ target)[0].reshape(d, rank)
        trace.append(env_error(env, left @ right.T))
        if len(trace) > 1 and abs(trace[-1]-trace[-2]) < 1e-12:
            break
    return left @ right.T, trace


def direct_check(model, table, length, seed):
    rng = np.random.default_rng(seed)
    vocab = model.embed.num_embeddings
    seq = torch.tensor(rng.integers(0, vocab, size=(16, length)), dtype=torch.long)
    with torch.no_grad():
        direct = model.layers[0](model.embed(seq)).numpy()
    table_outputs = np.zeros((16, length, model.embed.embedding_dim))
    arr = seq.numpy()
    for n in range(16):
        for i in range(length):
            for j in range(i+1):
                table_outputs[n, i] += table[i,j, arr[n,i], arr[n,j]]
    original = model.embed(seq).detach().numpy()
    predicted = (1-model.layers[0].scale)*original + model.layers[0].scale*table_outputs
    return float(np.max(np.abs(predicted-direct)))


def sequence_logit_error(model, table, original, seq):
    """Relative squared logit error on sequences not used by the decomposition."""
    arr = seq.numpy()
    batch, length = arr.shape
    out = np.zeros((batch, length, model.embed.embedding_dim))
    for n in range(batch):
        for i in range(length):
            for j in range(i+1):
                out[n, i] += table[i,j,arr[n,i],arr[n,j]]
    residual = model.embed(seq).detach().numpy()
    patched = (1-model.layers[0].scale)*residual + model.layers[0].scale*out
    logits = patched @ model.head.weight.detach().numpy().T
    return float(np.mean((logits-original)**2)/np.mean(original**2))


def collapsed_svd(table, measure, ranks):
    """Unconstrained functional SVD with zero completion off the token domain.

    Rows are (query position, query token), columns are (key position, key
    token, output). This compresses the *whole function*, unlike the QK1-bond
    methods, so rank numbers are descriptive rather than budget matched.
    """
    length, _, vocab, _, width = table.shape
    completed = table.copy()
    for i in range(length):
        invalid = ~np.eye(vocab,dtype=bool)
        completed[i,i,invalid] = 0
    mat = completed.transpose(0,2,1,3,4).reshape(length*vocab,length*vocab*width)
    gram = mat @ mat.T
    vals, vecs = sla.eigh(gram,subset_by_index=[gram.shape[0]-max(ranks),gram.shape[0]-1])
    vecs = vecs[:, ::-1]
    denom = np.sum(measure[...,None]*table**2)
    out = {}
    for rank in ranks:
        rebuilt = (vecs[:,:rank] @ (vecs[:,:rank].T @ mat)).reshape(length,vocab,length,vocab,width)
        rebuilt = rebuilt.transpose(0,2,1,3,4)
        error = float(np.sum(measure[...,None]*(rebuilt-table)**2)/denom)
        out[rank] = (rebuilt,error)
    return out


def bridge_control():
    rng = np.random.default_rng(156)
    n,d,o = 7,4,3
    u = rng.normal(size=(1,n,d))
    v = rng.normal(size=(1,n,d))
    a = rng.uniform(0.5,1.5,size=n)
    b = rng.uniform(0.5,1.5,size=n)
    second = np.outer(a,b)[None,None]
    c = rng.normal(size=(n,o))
    env = gram_environment(u,v,second,c)
    wtg, _, cycle, _, _ = weighted_trace_gauge(env,2)
    fet, _ = fet_als(env,wtg,2)
    return {"cycle_entropy_bits":cycle,"wtg_error":env_error(env,wtg),
            "fet_error":env_error(env,fet),"gap":env_error(env,wtg)-env_error(env,fet)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--length", type=int, default=8)
    p.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8, 12])
    p.add_argument("--head", type=int, default=0)
    p.add_argument("--zipf-alpha", type=float, default=0.0,
                   help="Token-probability exponent for the causal sequence metric.")
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    args = p.parse_args()
    torch.set_num_threads(1)
    model = load_model(args.checkpoint, args.length)
    assert 0 <= args.head < model.layers[0].n_head
    u, v, b, c = factors(model, args.length,head=args.head)
    d = v.shape[-1]
    assert all(0 < rank < d for rank in args.ranks)
    token_probs = np.arange(1,model.embed.num_embeddings+1,dtype=float)**(-args.zipf_alpha)
    token_probs /= token_probs.sum()
    measure = pair_measure(args.length,token_probs)
    env = gram_environment(u, v, b, c, measure)
    alternate_alpha = 1.0 if args.zipf_alpha == 0 else 0.0
    alternate_probs = np.arange(1,model.embed.num_embeddings+1,dtype=float)**(-alternate_alpha)
    alternate_probs /= alternate_probs.sum()
    alternate_measure = pair_measure(args.length,alternate_probs)
    alternate_env = gram_environment(u,v,b,c,alternate_measure)
    identity = np.eye(d)
    table = token_table(u,v,b,c,identity)
    background = np.zeros_like(table)
    for other in range(model.layers[0].n_head):
        if other != args.head:
            ou,ov,ob,oc = factors(model,args.length,head=other)
            background += token_table(ou,ov,ob,oc,np.eye(d))
    swap_u,swap_v,swap_b,swap_c = factors(model,args.length,head=args.head,swap=True)
    swapped_table = token_table(swap_u,swap_v,swap_b,swap_c,identity)
    branch_swap_max = float(np.max(np.abs(swapped_table-table)))
    assert branch_swap_max < 1e-9
    swap_env = gram_environment(swap_u,swap_v,swap_b,swap_c,measure)
    direct_max = direct_check(model,table+background,args.length,2878)
    assert direct_max < 1e-9, direct_max
    seed = np.random.default_rng(27)
    gauged = seed.standard_normal((d,d))
    gauged = sla.expm(0.15*(gauged-gauged.T)) @ np.diag(np.geomspace(0.5,2,d))
    ug = u @ gauged
    vg = v @ sla.inv(gauged).T
    gauge_max = float(np.max(np.abs(token_table(ug,vg,b,c,identity)-table)))
    assert gauge_max < 1e-9, gauge_max
    env_g = gram_environment(ug,vg,b,c,measure)
    seq = torch.tensor(seed.choice(model.embed.num_embeddings,size=(128,args.length),p=token_probs),dtype=torch.long)
    with torch.no_grad():
        original_logits = model(seq).numpy()
    alternate_seq = torch.tensor(seed.choice(model.embed.num_embeddings,size=(128,args.length),p=alternate_probs),dtype=torch.long)
    with torch.no_grad():
        alternate_logits = model(alternate_seq).numpy()
    exact_logit_error = sequence_logit_error(model,table+background,original_logits,seq)
    assert exact_logit_error < 1e-20
    bridge = bridge_control()
    assert abs(bridge["cycle_entropy_bits"]) < 1e-9
    assert abs(bridge["gap"]) < 1e-8
    collapsed = collapsed_svd(table,measure,args.ranks)
    intervention_b = np.broadcast_to(np.tril(np.ones((args.length,args.length)))[:,:,None,None],b.shape)
    intervention_target = token_table(u,v,intervention_b,c,np.eye(d))
    records = []
    arrays = {}
    _, singular, cycle, x, y = weighted_trace_gauge(env,max(args.ranks))
    _, singular_g, cycle_g, x_g, y_g = weighted_trace_gauge(env_g,max(args.ranks))
    _, swap_singular, swap_cycle, swap_x, swap_y = weighted_trace_gauge(swap_env,max(args.ranks))
    wtg_reconstruction, wtg_boundary_defect = wtg_boundary_check(u,v,b,c,measure,x,y,singular)
    for rank in args.ranks:
        local = local_svd(u,v,rank)
        wtg = cut_wtg(x,y,singular,rank)
        wtg_g = cut_wtg(x_g,y_g,singular_g,rank)
        swap_wtg = cut_wtg(swap_x,swap_y,swap_singular,rank)
        fet, trace = fet_als(env,wtg,rank)
        wtg_table = token_table(u,v,b,c,wtg)
        wtg_g_table = token_table(ug,vg,b,c,wtg_g)
        wtg_gauge_max = float(np.max(np.abs(wtg_table-wtg_g_table)))
        norm_singular = singular/np.linalg.norm(singular)
        norm_singular_g = singular_g/np.linalg.norm(singular_g)
        local_table=token_table(u,v,b,c,local)
        fet_table=token_table(u,v,b,c,fet)
        collapsed_table, collapsed_error = collapsed[rank]
        for name,bond,tab in (("local",local,local_table),("wtg",wtg,wtg_table),("fet",fet,fet_table)):
            np.testing.assert_allclose(env_error(env,bond),literal_error(table,tab,measure),atol=1e-10,rtol=1e-8)
            arrays[f"{name}_bond_rank{rank}"] = bond
        records.append({
            "rank":rank,"cycle_entropy_bits":cycle,
            "local_relative_mse":env_error(env,local),
            "wtg_relative_mse":env_error(env,wtg),
            "fet_relative_mse":env_error(env,fet),
            "collapsed_relative_mse":collapsed_error,
            "collapsed_factor_parameters":rank*(args.length*model.embed.num_embeddings)*(1+model.embed.embedding_dim),
            "local_sequence_logit_mse":sequence_logit_error(model,local_table+background,original_logits,seq),
            "wtg_sequence_logit_mse":sequence_logit_error(model,wtg_table+background,original_logits,seq),
            "fet_sequence_logit_mse":sequence_logit_error(model,fet_table+background,original_logits,seq),
            "collapsed_sequence_logit_mse":sequence_logit_error(model,collapsed_table+background,original_logits,seq),
            "local_alternate_metric_mse":env_error(alternate_env,local),
            "wtg_alternate_metric_mse":env_error(alternate_env,wtg),
            "fet_alternate_metric_mse":env_error(alternate_env,fet),
            "local_alternate_logit_mse":sequence_logit_error(model,local_table+background,alternate_logits,alternate_seq),
            "wtg_alternate_logit_mse":sequence_logit_error(model,wtg_table+background,alternate_logits,alternate_seq),
            "fet_alternate_logit_mse":sequence_logit_error(model,fet_table+background,alternate_logits,alternate_seq),
            "local_qk2_constant_intervention_mse":literal_error(intervention_target,token_table(u,v,intervention_b,c,local),measure),
            "wtg_qk2_constant_intervention_mse":literal_error(intervention_target,token_table(u,v,intervention_b,c,wtg),measure),
            "fet_qk2_constant_intervention_mse":literal_error(intervention_target,token_table(u,v,intervention_b,c,fet),measure),
            "wtg_gauge_max_abs":wtg_gauge_max,
            "wtg_full_reconstruction_max_abs":wtg_reconstruction,
            "wtg_boundary_defect":wtg_boundary_defect,
            "wtg_normalized_coefficients_max_abs":float(np.max(np.abs(norm_singular-norm_singular_g))),
            "cycle_entropy_gauge_abs":abs(cycle-cycle_g),
            "branch_swap_wtg_relative_mse":env_error(swap_env,swap_wtg),
            "fet_iterations":len(trace),
            "fet_step_converged":len(trace)>1 and abs(trace[-1]-trace[-2])<1e-12,
            "fet_trace":trace,
        })
        assert env_error(env,fet) <= env_error(env,wtg)+1e-9
        assert wtg_gauge_max < 1e-7, wtg_gauge_max
        assert records[-1]["wtg_normalized_coefficients_max_abs"] < 1e-8
        assert wtg_reconstruction < 1e-8 and wtg_boundary_defect < 1e-8
    arrays["wtg_x"], arrays["wtg_y"], arrays["wtg_coefficients"] = x,y,singular
    result={"checkpoint":str(args.checkpoint),"checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "length":args.length,"vocab":model.embed.num_embeddings,"head_dim":d,
            "head":args.head,"n_head":model.layers[0].n_head,
            "zipf_alpha":args.zipf_alpha,"metric":"realizable causal token-pair distribution",
            "alternate_metric_zipf_alpha":alternate_alpha,
            "branch_swap_table_max_abs":branch_swap_max,
            "branch_swap_cycle_entropy_bits":swap_cycle,
            "token_table_direct_max_abs":direct_max,"exact_gauge_max_abs":gauge_max,
            "exact_sequence_logit_mse":exact_logit_error,"bridge_control":bridge,"records":records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(args.output.with_suffix(".npz"),**arrays)
    print(json.dumps({k:v for k,v in result.items() if k != "records"},indent=2))
    for record in records:
        print({k:v for k,v in record.items() if k != "fet_trace"},flush=True)


if __name__ == "__main__":
    main()

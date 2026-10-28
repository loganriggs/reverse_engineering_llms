"""Load and evaluate compatible tensor-transformer checkpoints by snapshot path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_bilin18_mlp_discriminators import MODEL_REPO
from run_bilin18_qkov_causal_edges import FullModel


def load_snapshot_model(snapshot: Path, device="cuda"):
    sys.path.insert(0, str(MODEL_REPO))
    import jacclust.tt_model as tt

    def native_einsum(first, second, equation):
        if "-> ... n_head seq_q seq_k" in equation or "-> ... n_head seq_q2 seq_k2" in equation:
            return torch.einsum("bqhd,bkhd->bhqk", first, second)
        if "n_head seq_q seq_k" in equation and "seq_k n_head d_head" in equation:
            return torch.einsum("bhqk,bkhd->bhqd", first, second)
        raise ValueError(f"unsupported checkpoint einsum: {equation}")

    tt.einsum = native_einsum
    config_dict = json.loads((snapshot / "config.json").read_text())
    config_dict.pop("step", None)
    config = tt.GPTConfig(**config_dict)
    state = torch.load(snapshot / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    embedding = torch.nn.Embedding(config.vocab_size, config.n_embd)
    embedding.load_state_dict({"weight": state["transformer.wte.weight"]})
    blocks = torch.nn.ModuleList()
    for layer in range(config.n_layer):
        block = tt.Block(config)
        prefix = f"transformer.h.{layer}."
        block.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)})
        blocks.append(block)
    lm_head = tt.CastedLinear(config.n_embd, config.vocab_size, bias=False)
    lm_head.load_state_dict({"weight": state["lm_head.weight"]})
    del state
    dtype = torch.float32 if device.startswith("cuda") else torch.float32
    return FullModel(
        embedding.to(device=device, dtype=dtype).eval(),
        blocks.to(device=device, dtype=dtype).eval(),
        lm_head.to(device=device, dtype=dtype).eval(),
        config,
        tt,
    )


@torch.inference_mode()
def layer_input(model, ids, target_layer):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x = x0; value_bus = None
    for block in model.blocks[:target_layer]:
        x, value_bus = block(x, value_bus, x0)
    block = model.blocks[target_layer]
    mixed = block.lambdas[0] * x + block.lambdas[1] * x0
    attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
    return F.rms_norm(mixed + attention, (dimension,))


@torch.inference_mode()
def collect_states_writes(model, layer, corpus_config, positions, batch_size=4):
    source = np.load(corpus_config["path"], mmap_mode="r")
    rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
    states = []; writes = []; block = model.blocks[layer]
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x = layer_input(model, ids, layer)[:, positions].float().flatten(0, 1)
        write = block.mlp.Down(block.mlp.Left(x) * block.mlp.Right(x))
        states.append(x); writes.append(write)
    return torch.cat(states), torch.cat(writes)


@torch.inference_mode()
def final_logits(model, ids, intervention_layer=None, position=-1, mean=None, basis=None):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x = x0; value_bus = None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention
        normalized = F.rms_norm(pre, (dimension,))
        write = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        if layer == intervention_layer and basis is not None:
            selected = write[:, position].float()
            centered = selected - mean
            projected = mean + (centered @ basis) @ basis.T
            write = write.clone(); write[:, position] = projected.to(write.dtype)
        x = pre + write
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, position]).float()
    return 30 * torch.tanh(raw / 30)

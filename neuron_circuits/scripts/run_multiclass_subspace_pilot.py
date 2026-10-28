#!/usr/bin/env python3
"""Multiclass pilot: raw neurons versus rotations versus learned subspaces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from neuron_circuits.self_bilinear import DeepSelfBilinear


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--input-mode", choices=("keys", "prototypes"), default="keys")
    parser.add_argument("--facts", type=int, default=200)
    parser.add_argument("--content-dim", type=int, default=64)
    parser.add_argument("--train-samples-per-cell", type=int, default=64)
    parser.add_argument("--confirmation-samples-per-cell", type=int, default=32)
    parser.add_argument("--noise", type=float, default=.5)
    parser.add_argument("--operations", type=int, default=8)
    parser.add_argument("--code-dim", type=int, default=16)
    parser.add_argument("--neurons", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=.75)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotations", type=int, default=20)
    parser.add_argument("--context-holdout", action="store_true", help="Fit and test counterfactual directions on disjoint semantic contexts")
    parser.add_argument("--output", type=Path, default=Path("artifacts/multiclass_clean_neurons/round1_pilot10"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_data(args):
    generator = torch.Generator().manual_seed(args.seed + 1000)
    labels = torch.arange(args.facts) % args.classes
    labels = labels[torch.randperm(args.facts, generator=generator)]
    if args.operations > args.classes:
        raise ValueError("Operations must not exceed classes when shifts are unique")
    shifts = torch.arange(args.operations)
    raw_codes = torch.randn(args.code_dim, args.operations, generator=generator)
    codes = torch.linalg.qr(raw_codes).Q.T if args.code_dim >= args.operations else F.normalize(raw_codes.T, dim=1)
    if args.input_mode == "prototypes":
        codes = codes * math.sqrt(args.code_dim)
        prototypes = F.normalize(torch.randn(args.classes, args.content_dim, generator=generator), dim=1) * math.sqrt(args.content_dim)
        def generate(samples_per_cell):
            inputs, targets, base_classes, operation_ids = [], [], [], []
            for operation in range(args.operations):
                for base_class in range(args.classes):
                    content = prototypes[base_class] + args.noise * torch.randn(samples_per_cell, args.content_dim, generator=generator)
                    inputs.append(torch.cat((content, codes[operation].expand(samples_per_cell, -1)), dim=1))
                    targets.extend([int((base_class + shifts[operation]) % args.classes)] * samples_per_cell)
                    base_classes.extend([base_class] * samples_per_cell); operation_ids.extend([operation] * samples_per_cell)
            return torch.cat(inputs).to(args.device), torch.tensor(targets,device=args.device), torch.tensor(base_classes,device=args.device), torch.tensor(operation_ids,device=args.device)
        training = generate(args.train_samples_per_cell)
        discovery_mask = torch.zeros(len(training[0]),dtype=torch.bool,device=args.device)
        validation_mask = torch.zeros_like(discovery_mask)
        for cell in range(args.classes * args.operations):
            start = cell * args.train_samples_per_cell
            split = start + round(args.train_samples_per_cell * .8)
            end = start + args.train_samples_per_cell
            discovery_mask[start:split] = True; validation_mask[split:end] = True
        return {
            "train": training,
            "discovery": tuple(tensor[discovery_mask] for tensor in training),
            "validation": tuple(tensor[validation_mask] for tensor in training),
            "confirmation": generate(args.confirmation_samples_per_cell),
        }, torch.arange(args.classes,device=args.device), shifts.to(args.device), codes.to(args.device)
    examples = {"train": [], "discovery": [], "validation": [], "confirmation": []}
    rng = random.Random(args.seed + 2000)
    for operation in range(args.operations):
        by_class = {label: [] for label in range(args.classes)}
        for key, label in enumerate(labels.tolist()): by_class[label].append(key)
        train_keys, heldout_keys = [], []
        for keys in by_class.values():
            rng.shuffle(keys)
            if operation == 0:
                train_keys.extend(keys)
            else:
                split = round(len(keys) * args.train_fraction)
                train_keys.extend(keys[:split]); heldout_keys.extend(keys[split:])
        rng.shuffle(train_keys); rng.shuffle(heldout_keys)
        discovery_split = round(len(train_keys) * .8)
        for key in train_keys[:discovery_split]: examples["discovery"].append((key, operation))
        for key in train_keys[discovery_split:]: examples["validation"].append((key, operation))
        for key in train_keys: examples["train"].append((key, operation))
        for key in heldout_keys: examples["confirmation"].append((key, operation))
    def encode(rows):
        inputs = torch.zeros(len(rows), args.input_width + args.code_dim)
        keys = torch.tensor([row[0] for row in rows], dtype=torch.long)
        operations = torch.tensor([row[1] for row in rows], dtype=torch.long)
        inputs[torch.arange(len(rows)), keys] = 1
        inputs[:, args.input_width:] = codes[operations]
        targets = (labels[keys] + shifts[operations]) % args.classes
        return inputs.to(args.device), targets.to(args.device), keys.to(args.device), operations.to(args.device)
    return {split: encode(rows) for split, rows in examples.items()}, labels.to(args.device), shifts.to(args.device), codes.to(args.device)


@torch.inference_mode()
def metrics(model, dataset):
    logits = model(dataset[0]); targets = dataset[1]
    prediction = logits.argmax(-1)
    topk = min(5, logits.shape[-1])
    distance = (prediction - targets).abs(); distance = torch.minimum(distance, logits.shape[-1] - distance)
    return {
        "accuracy": float((prediction == targets).float().mean()),
        "top5_accuracy": float((logits.topk(topk, dim=1).indices == targets[:, None]).any(1).float().mean()),
        "cross_entropy": float(F.cross_entropy(logits, targets)),
        "mean_cyclic_error": float(distance.float().mean()),
    }


def train(args, data):
    torch.manual_seed(args.seed)
    model = DeepSelfBilinear(args.input_width + args.code_dim, args.neurons, args.depth, outputs=args.classes, residual_stream=True, decoder_bias=True).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    checks = 0; history = []; started = time.perf_counter()
    train_inputs, train_targets = data["train"][:2]
    for step in range(args.max_steps + 1):
        chosen = torch.randint(len(train_inputs), (args.batch_size,), device=args.device)
        logits = model(train_inputs[chosen]); loss = F.cross_entropy(logits, train_targets[chosen])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % 100 == 0:
            with torch.inference_mode():
                train_accuracy = float((model(train_inputs).argmax(-1) == train_targets).float().mean())
                train_loss = float(F.cross_entropy(model(train_inputs), train_targets))
            history.append({"step":step,"loss":train_loss,"accuracy":train_accuracy})
            checks = checks + 1 if train_loss < 1e-3 and train_accuracy > .999 else 0
            if checks >= 3: break
    return model, {"converged":checks>=3,"steps":step,"history":history,"elapsed_seconds":time.perf_counter()-started}


def operation_pairs(args, dataset, shifts, codes):
    inputs, targets, _, operations = dataset
    keep = operations != 0
    donor_inputs = inputs[keep]; donor_targets = targets[keep]; donor_operations = operations[keep]
    base_inputs = donor_inputs.clone()
    # Only the instruction changes: the content coordinates are identical.
    base_inputs[:, args.input_width:] = codes[0]
    base_targets = (donor_targets - shifts[donor_operations]) % args.classes
    return base_inputs, base_targets, donor_inputs, donor_targets


def class_pairs(args, dataset):
    """Pair independent examples with the same operation and adjacent classes."""
    inputs, targets, base_classes, operations = dataset
    base_indices, donor_indices = [], []
    for operation in range(args.operations):
        for base_class in range(args.classes):
            donor_class = (base_class + 1) % args.classes
            base = torch.nonzero((operations == operation) & (base_classes == base_class)).flatten()
            donor = torch.nonzero((operations == operation) & (base_classes == donor_class)).flatten()
            count = min(len(base), len(donor))
            base_indices.append(base[:count]); donor_indices.append(donor[:count])
    base_indices = torch.cat(base_indices); donor_indices = torch.cat(donor_indices)
    return inputs[base_indices], targets[base_indices], inputs[donor_indices], targets[donor_indices]


def filter_dataset(dataset, mask):
    return tuple(tensor[mask] for tensor in dataset)


@torch.inference_mode()
def layer_activations(model, inputs):
    return model(inputs, return_activations=True)[1]


def patched_logits(model, base_inputs, donor_inputs, layer_index, projector):
    with torch.inference_mode(): donor_activation = layer_activations(model, donor_inputs)[layer_index]
    hidden = base_inputs
    for index, layer in enumerate(model.layers):
        output, activation = layer(hidden)
        if index == layer_index:
            delta = donor_activation - activation
            patched = activation + delta @ projector
            output = F.linear(patched, layer.decoder.weight / math.sqrt(layer.neurons), layer.decoder.bias)
            if layer.residual: output = hidden + output
        hidden = output
    return model.readout(hidden)


@torch.inference_mode()
def patch_score(model, pair, layer_index, projector, temperature):
    base_inputs, base_targets, donor_inputs, donor_targets = pair
    base_logits = model(base_inputs); donor_logits = model(donor_inputs)
    patch_logits = patched_logits(model, base_inputs, donor_inputs, layer_index, projector)
    def contrast(logits):
        return logits.gather(1, donor_targets[:,None]).squeeze() - logits.gather(1, base_targets[:,None]).squeeze()
    base_contrast, donor_contrast, patch_contrast = map(contrast, (base_logits, donor_logits, patch_logits))
    denominator = donor_contrast - base_contrast
    valid = denominator > 1e-6
    recovery = (patch_contrast - base_contrast) / denominator.clamp_min(1e-6)
    donor_probability = F.softmax(donor_logits / temperature, dim=-1)
    def kl_to_donor(logits):
        return F.kl_div(F.log_softmax(logits/temperature,dim=-1), donor_probability, reduction="batchmean")
    base_kl = kl_to_donor(base_logits); patch_kl = kl_to_donor(patch_logits)
    return {
        "donor_target_accuracy": float((patch_logits.argmax(-1) == donor_targets).float().mean()),
        "mean_contrast_recovery": float(recovery[valid].mean()) if bool(valid.any()) else None,
        "median_contrast_recovery": float(recovery[valid].median()) if bool(valid.any()) else None,
        "kl_recovery": float(1 - patch_kl / base_kl.clamp_min(1e-12)),
        "valid_fraction": float(valid.float().mean()),
    }


def fit_projectors(model, pair, layer_index, rotations, generator):
    base, _, donor, _ = pair
    with torch.inference_mode():
        delta = layer_activations(model, donor)[layer_index] - layer_activations(model, base)[layer_index]
    covariance = delta.T @ delta / len(delta)
    _, _, vh = torch.linalg.svd(delta, full_matrices=False)
    raw_order = covariance.diag().argsort(descending=True)
    rotation_bases = [torch.linalg.qr(torch.randn(model.neurons, model.neurons, generator=generator, device=delta.device)).Q for _ in range(rotations)]
    rotation_orders = [(basis.T @ covariance @ basis).diag().argsort(descending=True) for basis in rotation_bases]
    return vh.T, raw_order, rotation_bases, rotation_orders


def projector_from_columns(columns):
    return columns @ columns.T


def analyze_family(args, model, discovery, validation, confirmation):
    with torch.inference_mode():
        logits = model(validation[2]); target = validation[3]
        correct = logits.gather(1,target[:,None]).squeeze(); masked=logits.clone(); masked.scatter_(1,target[:,None],-torch.inf)
        median_margin = (correct-masked.max(1).values).median().clamp_min(1e-3)
        temperature = float(median_margin / 5)
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 5000)
    output = {"temperature":temperature,"layers":{}}
    ranks = sorted({min(rank, args.neurons) for rank in (1, 2, 4, 8, 16, 32, 64, 128, args.neurons)})
    for layer in range(args.depth):
        pca, raw_order, rotation_bases, rotation_orders = fit_projectors(model, discovery, layer, args.rotations, generator)
        layer_rows = []
        for rank in ranks:
            rank = min(rank,args.neurons)
            raw_columns = torch.eye(args.neurons,device=args.device)[:,raw_order[:rank]]
            row = {"rank":rank,"raw_validation":patch_score(model,validation,layer,projector_from_columns(raw_columns),temperature),"raw_confirmation":patch_score(model,confirmation,layer,projector_from_columns(raw_columns),temperature),"pca_validation":patch_score(model,validation,layer,projector_from_columns(pca[:,:rank]),temperature),"pca_confirmation":patch_score(model,confirmation,layer,projector_from_columns(pca[:,:rank]),temperature)}
            rotation_validation=[]; rotation_confirmation=[]
            for basis,order in zip(rotation_bases,rotation_orders):
                projector=projector_from_columns(basis[:,order[:rank]])
                rotation_validation.append(patch_score(model,validation,layer,projector,temperature))
                rotation_confirmation.append(patch_score(model,confirmation,layer,projector,temperature))
            for split,values in (("validation",rotation_validation),("confirmation",rotation_confirmation)):
                row[f"rotation_{split}"]={metric:{"mean":statistics.mean(v[metric] for v in values),"sd":statistics.stdev(v[metric] for v in values)} for metric in ("donor_target_accuracy","mean_contrast_recovery","kl_recovery")}
            layer_rows.append(row)
        output["layers"][str(layer+1)] = layer_rows
    return output


def analyze(args, model, data, shifts, codes):
    if args.context_holdout:
        class_cut_1, class_cut_2 = args.classes // 2, 3 * args.classes // 4
        operation_cut_1, operation_cut_2 = args.operations // 2, 3 * args.operations // 4
        operation_sources = {
            "discovery": filter_dataset(data["discovery"], data["discovery"][2] < class_cut_1),
            "validation": filter_dataset(data["validation"], (data["validation"][2] >= class_cut_1) & (data["validation"][2] < class_cut_2)),
            "confirmation": filter_dataset(data["confirmation"], data["confirmation"][2] >= class_cut_2),
        }
        class_sources = {
            "discovery": filter_dataset(data["discovery"], data["discovery"][3] < operation_cut_1),
            "validation": filter_dataset(data["validation"], (data["validation"][3] >= operation_cut_1) & (data["validation"][3] < operation_cut_2)),
            "confirmation": filter_dataset(data["confirmation"], data["confirmation"][3] >= operation_cut_2),
        }
    else:
        operation_sources = class_sources = data
    families = {
        "operation": {
            split: operation_pairs(args, operation_sources[split], shifts, codes)
            for split in ("discovery", "validation", "confirmation")
        }
    }
    if args.input_mode == "prototypes":
        families["class"] = {
            split: class_pairs(args, class_sources[split])
            for split in ("discovery", "validation", "confirmation")
        }
    return {
        "families": {
            name: analyze_family(args, model, pairs["discovery"], pairs["validation"], pairs["confirmation"])
            for name, pairs in families.items()
        }
    }


def plot_analysis(args,analysis,path):
    families = analysis["families"]
    fig,axes=plt.subplots(len(families),args.depth,figsize=(6*args.depth,4*len(families)),sharey=True,squeeze=False)
    for row_index,(family,family_analysis) in enumerate(families.items()):
        for axis,(layer,rows) in zip(axes[row_index],family_analysis["layers"].items()):
            ranks=[r["rank"] for r in rows]
            axis.plot(ranks,[r["raw_confirmation"]["kl_recovery"] for r in rows],marker="o",label="raw neurons")
            axis.plot(ranks,[r["rotation_confirmation"]["kl_recovery"]["mean"] for r in rows],marker="o",label="rotated directions")
            axis.fill_between(ranks,[r["rotation_confirmation"]["kl_recovery"]["mean"]-r["rotation_confirmation"]["kl_recovery"]["sd"] for r in rows],[r["rotation_confirmation"]["kl_recovery"]["mean"]+r["rotation_confirmation"]["kl_recovery"]["sd"] for r in rows],alpha=.2)
            axis.plot(ranks,[r["pca_confirmation"]["kl_recovery"] for r in rows],marker="o",label="learned PCA subspace")
            axis.set_xscale("log",base=2); axis.set_ylim(-.1,1.05); axis.set_title(f"{family} counterfactual, layer {layer}"); axis.set_xlabel("patched dimensions"); axis.grid(alpha=.25)
        axes[row_index,0].set_ylabel("donor KL recovery")
    axes[0,-1].legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def main():
    args=parse_args(); args.input_width = args.facts if args.input_mode == "keys" else args.content_dim; args.output.mkdir(parents=True,exist_ok=True)
    data,labels,shifts,codes=build_data(args)
    model,training=train(args,data); model.eval()
    evaluations={split:metrics(model,dataset) for split,dataset in data.items()}
    analysis=analyze(args,model,data,shifts,codes)
    output={"round":"pilot","config":vars(args)|{"output":str(args.output)},"training":training,"evaluations":evaluations,"analysis":analysis,"shifts":shifts.tolist()}
    (args.output/"results.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"config":output["config"],"labels":labels.cpu(),"shifts":shifts.cpu(),"codes":codes.cpu()},args.output/"model.pt")
    plot_analysis(args,analysis,args.output/"basis_comparison.png")
    print(json.dumps({"training":training,"evaluations":evaluations},indent=2))


if __name__=="__main__": main()

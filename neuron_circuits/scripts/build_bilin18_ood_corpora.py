#!/usr/bin/env python3
"""Build deterministic GPT-2-tokenized OOD panels for Bilin18 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER


DATASETS = {
    "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "test"),
    "tinystories": ("roneneldan/TinyStories", None, "validation"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--sequence-length", type=int, default=129)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/bilin18_ood_corpora"))
    return parser.parse_args()


def build_rows(stream, tokenizer, rows, length):
    buffer, output = [], []
    eos = tokenizer.eos_token_id
    documents = 0
    for record in stream:
        text = record.get("text", "")
        if not text.strip():
            continue
        documents += 1
        buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        if eos is not None:
            buffer.append(eos)
        while len(buffer) >= length and len(output) < rows:
            output.append(buffer[:length])
            del buffer[:length]
        if len(output) == rows:
            break
    if len(output) < rows:
        raise RuntimeError(f"only built {len(output)} of {rows} rows")
    return np.asarray(output, dtype=np.int32), documents


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    manifest = {}
    for short, (name, config, split) in DATASETS.items():
        stream = load_dataset(name, config, split=split, streaming=True)
        array, documents = build_rows(stream, tokenizer, args.rows, args.sequence_length)
        path = args.output / f"{short}_{args.rows}x{args.sequence_length}.npy"
        np.save(path, array)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[short] = {
            "dataset": name, "config": config, "split": split,
            "documents_consumed": documents, "shape": list(array.shape),
            "path": str(path), "sha256": digest,
        }
        print(short, manifest[short], flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

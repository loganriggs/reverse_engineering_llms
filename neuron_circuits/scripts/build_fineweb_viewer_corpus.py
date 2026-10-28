#!/usr/bin/env python3
"""Build a deterministic fresh FineWeb token corpus for the final neuron viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


TOKENIZER = Path(
    "/home/loganriggs/.cache/huggingface/hub/models--gpt2/"
    "snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--sequence-length", type=int, default=513)
    parser.add_argument("--start-document", type=int, default=50_000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/bilin18_layers_0_2/corpus/fineweb_viewer_20k.npy"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True,
    )
    buffer, rows, texts = [], [], []
    document = 0

    def tokenize_pending():
        nonlocal buffer, texts
        if not texts:
            return
        for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
            buffer.extend(ids)
            buffer.append(tokenizer.eos_token_id)
        texts = []

    for example in dataset:
        document += 1
        if document < args.start_document:
            continue
        texts.append(example["text"])
        if len(texts) >= 256:
            tokenize_pending()
            while len(buffer) >= args.sequence_length and len(rows) < args.rows:
                rows.append(buffer[:args.sequence_length])
                del buffer[:args.sequence_length]
            if len(rows) >= args.rows:
                break
        if document % 5000 == 0:
            print(f"documents={document} rows={len(rows)}", flush=True)
    tokenize_pending()
    while len(buffer) >= args.sequence_length and len(rows) < args.rows:
        rows.append(buffer[:args.sequence_length])
        del buffer[:args.sequence_length]
    if len(rows) != args.rows:
        raise RuntimeError(f"requested {args.rows} rows, produced {len(rows)}")
    array = np.asarray(rows, dtype=np.uint16)
    if int(array.max()) > tokenizer.vocab_size - 1:
        raise ValueError("token id exceeds GPT-2 vocabulary")
    np.save(args.output, array)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "dataset": "HuggingFaceFW/fineweb",
        "configuration": "sample-10BT",
        "split": "train",
        "start_document_one_based": args.start_document,
        "end_document_one_based": document,
        "rows": args.rows,
        "sequence_length": args.sequence_length,
        "tokens": args.rows * args.sequence_length,
        "tokenizer": "gpt2",
        "tokenizer_snapshot": str(TOKENIZER),
        "dtype": "uint16",
        "sha256": digest,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

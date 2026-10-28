#!/usr/bin/env python3
"""Build token-level activation traces for natural sparse-neuron examples."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.hf_adapter import GatedMLPMaskAdapter


SOURCE = Path("artifacts/natural_neuron_examples/examples.json")
OUTPUT = Path("artifacts/token_activation_samples/data.json")
NEW_TOKENS = 12


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    model_name = source["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    adapter = GatedMLPMaskAdapter(model)
    continuation_cache = {}

    def continuation(prompt):
        if prompt not in continuation_cache:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                full_ids = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]
            continuation_cache[prompt] = (input_ids.shape[1], full_ids)
        return continuation_cache[prompt]

    output = {
        "model": model_name,
        "new_tokens": NEW_TOKENS,
        "families": {},
    }
    for family, family_node in source["families"].items():
        family_output = []
        for neuron in family_node["neurons"]:
            samples = []
            for polarity, rows in (("positive", neuron["top_positive"]), ("negative", neuron["top_negative"])):
                for rank, row in enumerate(rows, 1):
                    prompt_length, full_ids = continuation(row["prompt"])
                    with torch.no_grad():
                        trace = adapter.forward(full_ids.unsqueeze(0))
                    activations = trace.activations[neuron["layer"]][0, :, neuron["channel"]].detach().float().cpu()
                    token_ids = full_ids.detach().cpu().tolist()
                    tokens = [
                        {
                            "text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                            "activation": round(float(activation), 6),
                            "generated": index >= prompt_length,
                            "index": index,
                        }
                        for index, (token_id, activation) in enumerate(zip(token_ids, activations))
                    ]
                    samples.append({
                        "polarity": polarity,
                        "rank": rank,
                        "example_id": row["example_id"],
                        "answer": row["answer"],
                        "model_correct": row["model_correct"],
                        "selection_activation": row["activation"],
                        "circuit_position": neuron["position"],
                        "prompt_length": prompt_length,
                        "tokens": tokens,
                    })
            family_output.append({
                "unit": neuron["unit"],
                "layer": neuron["layer"],
                "position": neuron["position"],
                "channel": neuron["channel"],
                "recurrence_pairs": neuron["pairs"],
                "positive_count": neuron["positive_count"],
                "negative_count": neuron["negative_count"],
                "examples_scored": neuron["examples_scored"],
                "samples": samples,
            })
        output["families"][family] = family_output
        print(f"built token traces for {family}", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()

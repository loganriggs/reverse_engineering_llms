#!/usr/bin/env python3
"""Build a compact static token-activation visualization."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path


SOURCE = Path("artifacts/token_activation_samples/data.json")
OUTPUT = Path("/home/loganriggs/Coding/neuron-circuit-visualizations/token-activation-samples.html")


def visible(text: str) -> str:
    if not text.strip():
        return "↵" if "\n" in text else "␠"
    return text.replace("\n", "↵")


def sample_markup(title: str, unit: dict, sample: dict) -> str:
    scale = max(abs(token["activation"]) for token in sample["tokens"])
    parts = [
        '<section class="tas-sample">',
        f'<h3>{escape(title)}</h3>',
        f'<div class="tas-meta text-small">{escape(unit["unit"])} · inspected position {unit["position"]} · answer {escape(str(sample["answer"]))}</div>',
        '<div class="tas-tokens" role="img" aria-label="Tokens colored by signed raw neuron activation">',
    ]
    continuation = False
    for token in sample["tokens"]:
        if token["generated"] and not continuation:
            parts.append('<div class="tas-continuation text-small">model continuation →</div>')
            continuation = True
        value = token["activation"]
        strength = min(58.0, 8.0 + 50.0 * abs(value) / max(scale, 1e-9))
        tone = "var(--red)" if value < 0 else "var(--blue)"
        focus = " tas-focus" if token["index"] == sample["circuit_position"] else ""
        text = visible(token["text"])
        label = f'token {token["index"]}, {text}, activation {value:.3f}'
        parts.append(
            f'<span class="tas-token{focus}" style="--tas-tone:{tone};--tas-mix:{strength:.1f}%" aria-label="{escape(label)}">'
            f'<span class="tas-text">{escape(text)}</span><span class="tas-value">{value:.2f}</span></span>'
        )
    parts += ["</div>", "</section>"]
    return "\n".join(parts)


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    selections = []
    for family, label in (("arithmetic", "Arithmetic"), ("brackets", "Brackets")):
        unit = data["families"][family][0]
        positive = next(sample for sample in unit["samples"] if sample["polarity"] == "positive")
        negative = next(sample for sample in unit["samples"] if sample["polarity"] == "negative")
        selections.extend(((f"{label}: maximum positive natural example", unit, positive), (f"{label}: maximum negative natural example", unit, negative)))

    style = """<section id="token-activation-samples">
  <style>
    #token-activation-samples { display:grid; gap:20px; color:var(--foreground); }
    #token-activation-samples .tas-sample { display:grid; gap:7px; padding-top:12px; border-top:1px solid var(--border); }
    #token-activation-samples .tas-sample:first-of-type { border-top:0; padding-top:0; }
    #token-activation-samples .tas-meta { color:var(--muted-foreground); }
    #token-activation-samples .tas-tokens { display:flex; flex-wrap:wrap; align-items:stretch; gap:5px 3px; }
    #token-activation-samples .tas-token { display:grid; grid-template-rows:minmax(27px,auto) auto; align-items:center; min-width:32px; padding:4px 5px; background:color-mix(in srgb,var(--tas-tone) var(--tas-mix),transparent); color:var(--foreground); }
    #token-activation-samples .tas-focus { box-shadow:inset 0 -3px 0 var(--foreground); }
    #token-activation-samples .tas-text { white-space:pre-wrap; overflow-wrap:anywhere; font-weight:500; }
    #token-activation-samples .tas-value { font-size:.72em; opacity:.8; font-variant-numeric:tabular-nums; }
    #token-activation-samples .tas-continuation { flex-basis:100%; display:flex; align-items:center; gap:8px; margin-top:5px; color:var(--muted-foreground); }
    #token-activation-samples .tas-continuation::after { content:""; flex:1; border-top:1px solid var(--border); }
    #token-activation-samples .tas-legend { display:flex; flex-wrap:wrap; align-items:center; gap:8px; color:var(--muted-foreground); }
    #token-activation-samples .tas-ramp { width:min(220px,50vw); height:12px; background:linear-gradient(90deg,color-mix(in srgb,var(--red) 58%,transparent),transparent 50%,color-mix(in srgb,var(--blue) 58%,transparent)); }
  </style>
  <h2>Raw neuron activation across tokens</h2>
  <div class="tas-legend text-small"><span>negative</span><span class="tas-ramp" aria-hidden="true"></span><span>positive</span><span>• underline = sparse-circuit position</span></div>"""
    ending = "\n</section>\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(style + "\n" + "\n".join(sample_markup(*selection) for selection in selections) + ending, encoding="utf-8")


if __name__ == "__main__":
    main()

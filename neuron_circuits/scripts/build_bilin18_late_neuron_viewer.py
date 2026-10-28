#!/usr/bin/env python3
"""Build a standalone layers 15--17 input/output neuron viewer."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("artifacts/bilin18_layers_15_17")
OUT = ROOT / "neuron_viewer"


def compact(value, digits=5):
    return round(float(value), digits)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output_raw = json.loads((ROOT / "output_geometry/results.json").read_text())
    output = {
        item["layer"]: {channel["unit"]: channel for channel in item["channels"]}
        for item in output_raw["layer_results"]
    }
    traces_raw = json.loads((ROOT / "source_traces/results.json").read_text())
    traces = {
        (layer["layer"], unit["unit"]): unit
        for layer in traces_raw["layers"] for unit in layer["units"]
    }
    compression = json.loads(
        Path("artifacts/bilin18_final_mlp_compression_replicate/results.json").read_text()
    )
    ranks = {unit: rank + 1 for rank, unit in enumerate(compression["ranking_order"])}
    layers = []
    for layer in (15, 16, 17):
        activation = json.loads((ROOT / f"viewer_atlas/layer_{layer}/results.json").read_text())
        channels = []
        for channel in activation["all_channels"]:
            unit = channel["unit"]
            out = output[layer][unit]
            merged = {
                **channel,
                "direct_logit_norm": out["direct_logit_norm"],
                "max_logit_fraction": out["max_logit_fraction"],
                "vocab_effective_support": out["vocab_effective_support"],
                "output_class": out["output_class"],
                "output_class_lift": out["output_class_lift"],
                "output_positive_class": out["output_positive_class"],
                "output_negative_class": out["output_negative_class"],
                "top_output_tokens": out["top_output_tokens"][:8],
                "top_output_values": out["top_output_values"][:8],
                "bottom_output_tokens": out["bottom_output_tokens"][:8],
                "bottom_output_values": out["bottom_output_values"][:8],
            }
            if layer == 17:
                merged["compression_rank"] = ranks[unit]
                merged["compression_score"] = compact(compression["channel_scores"][unit])
            trace = traces.get((layer, unit))
            if trace:
                merged["trace"] = trace
            channels.append(merged)
        summary = {key: value for key, value in activation.items() if not isinstance(value, (list, dict))}
        summary["token_classes"] = activation["token_classes"]
        summary["output"] = next(
            x["aggregate"] for x in output_raw["layer_results"] if x["layer"] == layer
        )
        layers.append({"layer": layer, "summary": summary, "channels": channels})
    curves = []
    for row in compression["conditions"]:
        if row["selection"] == "ranked" and row["replacement"] == "mean":
            curves.append({
                "k": row["k"], "ce_delta": compact(row["cross_entropy_delta"]),
                "kl": compact(row["mean_kl_from_full"]),
                "top1": compact(row["top1_agreement"]),
            })
    data = {
        "layers": layers,
        "compression": {
            "baseline_ce": compression["baseline"]["cross_entropy"],
            "effective_support": compression["score_effective_support"],
            "score_fraction_top_k": compression["score_fraction_top_k"],
            "curves": curves,
            "eval_tokens": compression["baseline"]["tokens"],
        },
    }
    (OUT / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    (OUT / "index.html").write_text(HTML)
    print(OUT / "index.html")
    print(OUT / "data.json")


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bilin18 late-layer neuron viewer</title>
<style>
:root{color-scheme:light dark;--bg:light-dark(#f7f7f4,#101210);--panel:light-dark(#fff,#191c19);--ink:light-dark(#171a17,#edf1ec);--muted:light-dark(#626862,#aab3aa);--line:light-dark(#d9ddd7,#343a34);--a:light-dark(#176b57,#62c5a7);--b:light-dark(#a7432e,#ef8c73);--c:light-dark(#3e61a8,#86a7ed);--good:light-dark(#245f35,#78cc8c)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif}.shell{max-width:1480px;margin:auto;padding:20px}h1{font-size:24px;margin:0}h2,h3,h4{margin:0 0 10px}.subtitle,.mini,.foot{color:var(--muted)}.subtitle{margin:4px 0 0}.top{display:flex;justify-content:space-between;align-items:end;gap:16px;flex-wrap:wrap}.controls{display:flex;gap:8px;align-items:end;flex-wrap:wrap}.control{display:grid;gap:3px}.control label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}button,input,select{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--line);padding:.5rem .65rem;min-height:38px}button{cursor:pointer}#unit{width:92px}.summaries{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:18px 0}.summary,.panel{background:var(--panel);border:1px solid var(--line);padding:14px}.summary strong{font-size:17px}.main{display:grid;grid-template-columns:minmax(420px,1fr) minmax(460px,1.2fr);gap:14px}.panel{min-width:0}.chart{width:100%;height:auto;border:1px solid var(--line);background:var(--panel)}.axis{stroke:var(--line)}.label{fill:var(--muted);font-size:11px}.dot{fill:var(--a);opacity:.18}.dot.filtered{display:none}.dot.core{fill:var(--c);opacity:.48}.dot.selected{fill:var(--b);opacity:1;stroke:var(--ink);stroke-width:1.5}.badge{display:inline-block;border:1px solid var(--line);padding:3px 7px;margin:0 5px 8px 0;color:var(--good)}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.metric{border-left:3px solid var(--a);padding:4px 8px}.metric span,.reader span{display:block;color:var(--muted);font-size:11px}.metric b{font-size:17px}.classes{margin:12px 0;padding:10px;background:color-mix(in srgb,var(--a) 7%,transparent)}.two{display:grid;grid-template-columns:1fr 1fr;gap:10px}.ctx{padding:7px 8px;border-left:2px solid var(--line);margin:6px 0;overflow-wrap:anywhere}.pos{border-color:var(--a)}.neg{border-color:var(--b)}.tokens{display:flex;gap:6px;flex-wrap:wrap}.token{border:1px solid var(--line);padding:3px 6px;font-family:ui-monospace,monospace}.trace{border-top:1px solid var(--line);padding-top:9px;margin-top:9px}.pair{font:12px ui-monospace,monospace;overflow-wrap:anywhere}.compression{margin-top:14px}.legend{display:flex;gap:14px;flex-wrap:wrap;margin:7px 0}.swatch{display:inline-block;width:10px;height:10px;margin-right:4px}.foot{margin-top:12px;font-size:12px}@media(max-width:900px){.main,.summaries,.two{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main class="shell"><div class="top"><div><h1>Bilin18 late-layer neuron viewer</h1><p class="subtitle">Layers 15–17 · input discrimination, unembedding role, and final-layer compression</p></div><div class="controls"><div class="control"><label for="layer">Layer</label><select id="layer"><option>17</option><option>16</option><option>15</option></select></div><div class="control"><label for="unit">Unit</label><input id="unit" type="number" min="0" max="4607" value="3504"></div><button id="prev">←</button><button id="next">→</button><div class="control"><label for="class">Input class</label><select id="class"><option value="all">all</option></select></div><button id="core">Compression core</button><button id="traced">Traced examples</button></div></div><div id="summaries" class="summaries"></div><div class="main"><section class="panel"><h2>Input–output map</h2><div class="mini">Held-out input lift versus direct output-class lift. Blue marks are the top 128 final-layer compression channels.</div><svg id="scatter" class="chart" viewBox="0 0 680 420" aria-label="Input and output class lift scatter plot"></svg><div class="compression"><h3>Can layer 17 be made smaller?</h3><svg id="curve" class="chart" viewBox="0 0 680 300" aria-label="Final MLP compression curves"></svg><div class="legend"><span><i class="swatch" style="background:var(--a)"></i>KL from full model</span><span><i class="swatch" style="background:var(--b)"></i>cross-entropy increase</span></div><div class="mini">Fixed channels selected on a disjoint discovery split; omitted activations are replaced by their discovery mean.</div></div></section><section id="detail" class="panel" aria-live="polite"></section></div></main>
<script>
let DATA;const state={layer:17,unit:3504,cls:'all'};const layerEl=document.getElementById('layer'),unitEl=document.getElementById('unit'),classEl=document.getElementById('class');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=v=>Number(v).toLocaleString(undefined,{maximumFractionDigits:2});
fetch('./data.json').then(r=>r.json()).then(d=>{DATA=d;init()}).catch(()=>document.getElementById('detail').innerHTML='<h2>Viewer data unavailable</h2><p>Serve this directory over HTTP so data.json can load.</p>');
function layer(){return DATA.layers.find(x=>x.layer===state.layer)}function channel(){return layer().channels[state.unit]}
function init(){for(const c of DATA.layers[0].summary.token_classes){const o=document.createElement('option');o.value=c;o.textContent=c;classEl.appendChild(o)}wire();renderSummaries();renderCurve();render()}
function commit(){state.unit=Math.max(0,Math.min(4607,+unitEl.value||0));unitEl.value=state.unit;render()}
function wire(){layerEl.onchange=()=>{state.layer=+layerEl.value;state.unit=state.layer===17?3504:0;unitEl.value=state.unit;render()};unitEl.onchange=commit;unitEl.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();commit()}};classEl.onchange=()=>{state.cls=classEl.value;renderScatter()};document.getElementById('prev').onclick=()=>step(-1);document.getElementById('next').onclick=()=>step(1);document.getElementById('core').onclick=()=>{state.layer=17;layerEl.value='17';const u=layer().channels.find(c=>c.compression_rank===1).unit;state.unit=u;unitEl.value=u;render()};document.getElementById('traced').onclick=()=>{const cs=layer().channels;let u=cs.findIndex((c,i)=>i>state.unit&&c.trace);if(u<0)u=cs.findIndex(c=>c.trace);if(u>=0){state.unit=u;unitEl.value=u;render()}}}
function step(d){state.unit=(state.unit+d+4608)%4608;unitEl.value=state.unit;render()}
function renderSummaries(){document.getElementById('summaries').innerHTML=DATA.layers.map(x=>`<div class="summary"><strong>Layer ${x.layer}</strong><div class="mini">median input lift <b>${fmt(x.summary.active_half_heldout_lift_median)}×</b> · signed gap <b>${fmt(x.summary.active_half_signed_gap_sd_median)} SD</b><br>${fmt(100*x.summary.active_half_fraction_heldout_lift_ge_2)}% ≥2× · ${fmt(100*x.summary.active_half_fraction_signed_gap_ge_1_sd)}% ≥1 SD</div></div>`).join('')}
function render(){renderScatter();renderDetail()}
function renderScatter(){const svg=document.getElementById('scatter'),cs=layer().channels,W=680,H=420,m={l:60,r:18,t:20,b:52};const xs=cs.map(c=>c.heldout_product_lift),ys=cs.map(c=>c.output_class_lift),xmax=Math.min(8,Math.max(...xs)),ymax=Math.min(4,Math.max(...ys));const x=v=>m.l+(Math.min(v,xmax)-1)/(xmax-1)*(W-m.l-m.r),y=v=>H-m.b-(Math.min(v,ymax)-1)/(ymax-1)*(H-m.t-m.b);let h=`<line class="axis" x1="${m.l}" y1="${H-m.b}" x2="${W-m.r}" y2="${H-m.b}"/><line class="axis" x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H-m.b}"/><text class="label" x="${W/2}" y="410" text-anchor="middle">held-out input-class lift (×)</text><text class="label" transform="translate(15 ${H/2}) rotate(-90)" text-anchor="middle">direct output-class lift (×)</text>`;for(const t of [1,2,4,8])if(t<=xmax)h+=`<text class="label" x="${x(t)}" y="${H-m.b+18}" text-anchor="middle">${t}</text>`;for(const t of [1,2,3,4])if(t<=ymax)h+=`<text class="label" x="${m.l-8}" y="${y(t)+4}" text-anchor="end">${t}</text>`;h+=cs.map(c=>`<circle class="dot ${state.cls!=='all'&&c.product_class!==state.cls?'filtered':''} ${state.layer===17&&c.compression_rank<=128?'core':''} ${c.unit===state.unit?'selected':''}" data-u="${c.unit}" cx="${x(c.heldout_product_lift)}" cy="${y(c.output_class_lift)}" r="${c.unit===state.unit?5:2.2}"/>`).join('');svg.innerHTML=h;svg.querySelectorAll('.dot').forEach(n=>n.onclick=()=>{state.unit=+n.dataset.u;unitEl.value=state.unit;render()})}
function renderCurve(){const svg=document.getElementById('curve'),rows=DATA.compression.curves.filter(r=>r.k>0),W=680,H=300,m={l:60,r:20,t:20,b:52},x=v=>m.l+Math.log2(v/8)/Math.log2(4608/8)*(W-m.l-m.r),max=.34,y=v=>H-m.b-Math.min(v,max)/max*(H-m.t-m.b);let h=`<line class="axis" x1="${m.l}" y1="${H-m.b}" x2="${W-m.r}" y2="${H-m.b}"/><line class="axis" x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H-m.b}"/><text class="label" x="${W/2}" y="292" text-anchor="middle">retained native channels (log scale)</text><text class="label" transform="translate(15 ${H/2}) rotate(-90)" text-anchor="middle">divergence / nats</text>`;for(const k of [8,32,128,512,2048,4608])h+=`<text class="label" x="${x(k)}" y="${H-m.b+18}" text-anchor="middle">${k}</text>`;for(const t of [0,.1,.2,.3])h+=`<text class="label" x="${m.l-8}" y="${y(t)+4}" text-anchor="end">${t.toFixed(1)}</text>`;for(const [key,color] of [['kl','var(--a)'],['ce_delta','var(--b)']]){const points=rows.map(r=>`${x(r.k)},${y(r[key])}`).join(' ');h+=`<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2"/>`;h+=rows.map(r=>`<circle cx="${x(r.k)}" cy="${y(r[key])}" r="3" fill="${color}"/>`).join('')}svg.innerHTML=h}
function contexts(c,key,klass){return (c[key+'_tail_contexts']||[]).map((s,i)=>`<div class="ctx ${klass}"><b>${fmt(c[key+'_tail_values'][i])}</b> · ${esc(s)}</div>`).join('')}
function toks(ts,vs){return ts.map((t,i)=>`<span class="token" title="${fmt(vs[i])}">${esc(JSON.stringify(t))}</span>`).join('')}
function trace(c){if(!c.trace)return '<h3>Named-source trace</h3><div class="mini">Not run for this unit.</div>';return '<h3>Named-source trace</h3>'+c.trace.sites.map(s=>`<div class="trace"><b>${esc(s.tail)}:</b> ${esc(s.context)}<div>product ${fmt(s.actual_product)} → ${fmt(s.without_local_attention_product)} without local attention</div><div>largest raw source pair ${fmt(100*s.dominant_pair_abs_fraction)}% · effective support ${fmt(s.pair_effective_support)}</div><div class="pair">${s.source_pairs.slice(0,5).map(p=>`${esc(p.first)}×${esc(p.second)} ${fmt(p.contribution)}`).join(' · ')}</div></div>`).join('')}
function renderDetail(){const c=channel(),badges=[];if(c.heldout_product_lift>=2&&c.heldout_signed_class_gap_sd>=1)badges.push('strong held-out input separator');if(state.layer===17&&c.compression_rank<=128)badges.push(`compression core rank ${c.compression_rank}`);if(c.trace)badges.push('exact source trace');document.getElementById('detail').innerHTML=`${badges.map(x=>`<span class="badge">${x}</span>`).join('')}<h2>Layer ${state.layer} · unit ${c.unit}</h2><div class="metrics"><div class="metric"><span>input lift</span><b>${fmt(c.heldout_product_lift)}×</b></div><div class="metric"><span>signed gap</span><b>${fmt(c.heldout_signed_class_gap_sd)} SD</b></div><div class="metric"><span>output lift</span><b>${fmt(c.output_class_lift)}×</b></div><div class="metric"><span>direct logit norm</span><b>${fmt(c.direct_logit_norm)}</b></div><div class="metric"><span>vocab support</span><b>${fmt(c.vocab_effective_support)}</b></div><div class="metric"><span>compression rank</span><b>${state.layer===17?c.compression_rank:'—'}</b></div></div><div class="classes"><b>Input product:</b> ${esc(c.product_positive_class)} → + / ${esc(c.product_negative_class)} → −; |product| class ${esc(c.product_class)}<br><b>Direct output:</b> ${esc(c.output_positive_class)} → + / ${esc(c.output_negative_class)} → −; magnitude class ${esc(c.output_class)}</div><h3>2.56M-token activation tails</h3><div class="two"><div><h4>Positive</h4>${contexts(c,'positive','pos')}</div><div><h4>Negative</h4>${contexts(c,'negative','neg')}</div></div><h3>Direct Down→unembedding tails</h3><div class="two"><div><h4>Promotes</h4><div class="tokens">${toks(c.top_output_tokens,c.top_output_values)}</div></div><div><h4>Suppresses</h4><div class="tokens">${toks(c.bottom_output_tokens,c.bottom_output_values)}</div></div></div>${trace(c)}<div class="foot">Input labels were frozen on 2,048,000 FineWeb positions and evaluated on 512,000 held-out positions. Output tokens are a raw logit lens before final RMSNorm; broad support is expected and is not a causal attribution.</div>`}
</script></body></html>'''


if __name__ == "__main__":
    main()

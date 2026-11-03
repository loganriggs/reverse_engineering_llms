"""Exploratory full versus local scores for pinned FineWeb targets."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from score_fineweb_prediction_sites import score, suffix_words


@torch.no_grad()
def run(preview_path, ledger_path, device, model_name, tensor_language_root):
    rows = {r['index']: r for r in json.loads(Path(preview_path).read_text())}
    ledger = json.loads(Path(ledger_path).read_text())
    tok = AutoTokenizer.from_pretrained('gpt2' if model_name == 'bilin18' else model_name)
    if model_name == 'bilin18':
        import sys
        root = Path(tensor_language_root) if tensor_language_root else (
            Path(__file__).resolve().parents[2] / 'reverse_engineer/tensor_language')
        sys.path[:0] = [str(root), str(root / 'basis_aligned/qk_mdl')]
        from tier2_model import load_elriggs, reference_forward
        model, _ = load_elriggs('bilin18', device=device, dtype=torch.float32)
        forward = lambda ids: reference_forward(model, ids)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        forward = lambda ids: model(ids).logits
    sites = []
    for site in ledger['sites']:
        row = rows[site['source_row']]
        assert row['sha256'] == site['source_sha256']
        prefix = row['preview'][:site['start_char']]
        assert row['preview'][site['start_char']:site['end_char']] == site['target']
        values = {name: score(forward, tok, device, context, site['target'])
                  for name, context in [('full', prefix),
                                        ('last_100_words', suffix_words(prefix, 100)),
                                        ('last_40_words', suffix_words(prefix, 40))]}
        alternative = site.get('alternative')
        alt_values = None
        if alternative:
            alt_values = {
                name: score(forward, tok, device, context, alternative)
                for name, context in [('full', prefix),
                                      ('last_100_words', suffix_words(prefix, 100)),
                                      ('last_40_words', suffix_words(prefix, 40))]}
        sites.append(dict(source_row=site['source_row'],
                          source_sha256=site['source_sha256'],
                          target=site['target'], mechanism=site['mechanism'],
                          predictability=site['predictability'], scores=values,
                          alternative=alternative, alternative_scores=alt_values))
        print(site['source_row'], site['target'],
              round(values['full']['logprob'] - values['last_40_words']['logprob'], 3),
              flush=True)
    return dict(model=model_name, source='FineWeb sample-10BT',
                selection=ledger['selection'], sites=sites,
                caveat='Selected words or phrases are not independent token positions. '
                       'A model score difference measures sensitivity to '
                       'prefix truncation, not the human writer’s mechanism '
                       'or a causal effect. Target/alternative scores may '
                       'have different token counts. One model and one '
                       'annotation seed.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--preview-json', type=Path, required=True)
    p.add_argument('--ledger-json', type=Path, required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--model', default='gpt2')
    p.add_argument('--tensor-language-root')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run(args.preview_json, args.ledger_json, args.device, args.model,
                 args.tensor_language_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n')

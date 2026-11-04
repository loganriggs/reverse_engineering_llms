"""Exploratory local-LM semantic filter for literal multi-field candidates.

No model judgment is a semantic certificate. Use this only to estimate
whether manual/high-precision validation can scale beyond lexical rules.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_structured_wikipedia_multifield import (
    candidate, clean_field, literal_mentions, walk_fields,
)
from audit_structured_wikipedia_section_sites import paragraphs


def candidates(parquet, *, max_rows=10000, seed=74701, per_source=30):
    import pyarrow.parquet as pq
    file = pq.ParquetFile(parquet)
    rng = random.Random(seed)
    groups = list(range(file.num_row_groups))
    rng.shuffle(groups)
    found = {'lead': [], 'later': []}
    scanned = 0
    for group in groups:
        rows = file.read_row_group(
            group, columns=['name', 'abstract', 'infoboxes', 'sections']
        ).to_pylist()
        rng.shuffle(rows)
        for row in rows[:min(256, len(rows))]:
            if scanned >= max_rows:
                break
            scanned += 1
            lead = candidate(row)
            if len(lead) >= 2 and len(found['lead']) < per_source:
                found['lead'].append((row['name'], row['abstract'], lead[:2]))
            if not row['sections'] or not row['infoboxes'] or \
                    len(found['later']) >= per_source:
                continue
            try:
                sections = json.loads(row['sections'])
                infobox = json.loads(row['infoboxes'])
            except ValueError:
                continue
            fields = [x for x in (clean_field(a, b)
                                  for a, b in walk_fields(infobox)) if x]
            for text in paragraphs(sections):
                if not 40 <= len(text.split()) <= 220:
                    continue
                matched = []
                for label, value in fields:
                    if (len(literal_mentions(text, value)) == 1 and
                            label.casefold() not in
                            [f['label'].casefold() for f in matched]):
                        matched.append({'label': label, 'value': value})
                    if len(matched) == 2:
                        break
                if len(matched) == 2:
                    found['later'].append((row['name'], text, matched))
                    break
        if scanned >= max_rows or all(len(v) >= per_source
                                      for v in found.values()):
            break
    return found, scanned


def prompt(title, paragraph, fields):
    facts = '\n'.join(f"{i+1}. {x['label']} = {x['value']}"
                      for i, x in enumerate(fields))
    system = ('Judge only what the supplied prose asserts about the article '
              'subject. A value appearing in a sentence about another '
              'person, team, species, or place does not establish the '
              'article subject\'s field. Do not use outside knowledge. '
              'Return one compact JSON object with keys fact1, fact2, '
              'and reason. fact1 and fact2 must be booleans. Answer true '
              'only when the prose asserts that the article subject has '
              'that exact field value. Be strict and use false for '
              'ambiguity. Do not judge whether counterfactual edits are safe.')
    user = (f'Article title: {title}\n'
            f'Paragraph: {paragraph}\n'
            f'Candidate fields:\n{facts}\nJSON:')
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': user}]


def parse_json(raw):
    start, end = raw.find('{'), raw.rfind('}')
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(raw[start:end+1])
    except ValueError:
        return None
    if not all(type(value.get(key)) is bool
               for key in ('fact1', 'fact2')):
        return None
    return value


def run(args):
    found, scanned = candidates(args.parquet, max_rows=args.max_rows,
                                seed=args.seed, per_source=args.per_source)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation='sdpa').to(args.device)
    model.eval()
    result = {'source': 'wikimedia/structured-wikipedia enwiki shard 0',
              'model': 'Qwen3-4B', 'seed': args.seed,
              'scanned': scanned, 'per_source': args.per_source,
              'counts': {}, 'decisions': [],
              'caveat': 'Automated judgments are uncalibrated hypotheses, '
                        'not semantic acceptance or training approval. '
                        'Only hashes and aggregate decisions are exported.'}
    counts = Counter()
    for source, examples in found.items():
        for title, paragraph, fields in examples:
            messages = prompt(title, paragraph, fields)
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            inputs = tokenizer(formatted, return_tensors='pt').to(args.device)
            with torch.no_grad():
                ids = model.generate(
                    **inputs, do_sample=False, max_new_tokens=130,
                    pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(ids[0][inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True)
            decision = parse_json(raw)
            counts[source+'/examples'] += 1
            counts[source+'/parsed'] += decision is not None
            if decision is not None:
                counts[source+'/both_asserted'] += (
                    decision['fact1'] and decision['fact2'])
            digest = hashlib.sha256(
                (title+'\n'+paragraph).encode()).hexdigest()[:16]
            result['decisions'].append({
                'source': source, 'document_hash': digest,
                'labels': [x['label'] for x in fields],
                'decision': {key: decision[key]
                             for key in ('fact1', 'fact2')}
                    if decision is not None else None})
            if counts[source+'/examples'] <= 5:
                print(source, title, [x['label'] for x in fields],
                      decision, flush=True)
    result['counts'] = dict(counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--max-rows', type=int, default=10000)
    parser.add_argument('--per-source', type=int, default=30)
    parser.add_argument('--seed', type=int, default=74701)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

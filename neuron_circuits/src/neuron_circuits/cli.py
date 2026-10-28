"""Command-line entry points for the pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from .tasks.relational import RelationalRetrievalGenerator, answer_counts, write_jsonl


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extraction-size", type=int, default=100)
    parser.add_argument("--transfer-per-example", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    generator = RelationalRetrievalGenerator(seed=args.seed)
    examples = generator.generate(args.extraction_size, args.transfer_per_example)
    write_jsonl(examples, args.output)
    extraction = [example for example in examples if example.split == "extraction"]
    print(f"wrote {len(examples)} examples to {args.output}")
    print(f"extraction answer counts: {dict(sorted(answer_counts(extraction).items()))}")


if __name__ == "__main__":
    main()

"""Minimal one-operand arithmetic counterfactuals."""

from __future__ import annotations

from dataclasses import dataclass

from neuron_circuits.arithmetic import ArithmeticExample, arithmetic_examples


@dataclass(frozen=True)
class ArithmeticCounterfactualPair:
    pair_id: str
    family: str
    base: ArithmeticExample
    source: ArithmeticExample
    change: str


def arithmetic_counterfactual_pairs() -> list[ArithmeticCounterfactualPair]:
    """Ordered pairs that preserve operation/left operand and change the right operand."""
    examples = arithmetic_examples()
    output = []
    for base in examples:
        for source in examples:
            if (
                base.operation != source.operation
                or base.left != source.left
                or base.right == source.right
                or base.answer == source.answer
            ):
                continue
            output.append(ArithmeticCounterfactualPair(
                pair_id=f"arith-{base.example_id}-to-{source.right}",
                family=base.operation,
                base=base,
                source=source,
                change=f"right operand {base.right} → {source.right}",
            ))
    return output

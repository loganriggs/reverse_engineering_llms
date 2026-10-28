from neuron_circuits.arithmetic_counterfactuals import arithmetic_counterfactual_pairs


def test_arithmetic_counterfactuals_change_one_operand_and_answer():
    pairs = arithmetic_counterfactual_pairs()
    assert pairs
    for pair in pairs:
        assert pair.base.operation == pair.source.operation
        assert pair.base.left == pair.source.left
        assert pair.base.right != pair.source.right
        assert pair.base.answer != pair.source.answer

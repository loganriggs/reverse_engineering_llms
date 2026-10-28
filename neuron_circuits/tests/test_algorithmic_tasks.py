from neuron_circuits.algorithmic_tasks import (
    algorithm_counterfactual_pairs,
    answer_for,
)


def test_algorithm_answers():
    values = (2, 8, 5, 1)
    assert answer_for(values, "maximum") == 8
    assert answer_for(values, "minimum") == 1
    assert answer_for(values, "second_largest") == 5
    assert answer_for(values, "argmax") == 1
    assert answer_for(values, "second_argmax") == 2


def test_counterfactual_pairs_change_one_value_and_answer():
    for algorithm in ("maximum", "minimum"):
        pairs = algorithm_counterfactual_pairs(algorithm)
        assert pairs
        for pair in pairs[:100]:
            differences = [
                index for index, (base, source) in enumerate(zip(pair.base.values, pair.source.values))
                if base != source
            ]
            assert differences == [pair.changed_position]
            assert pair.base.answer != pair.source.answer

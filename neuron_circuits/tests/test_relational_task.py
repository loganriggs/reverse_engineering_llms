import unittest

from neuron_circuits.tasks.relational import (
    PAYLOADS,
    RelationalRetrievalGenerator,
    answer_counts,
)


class RelationalTaskTests(unittest.TestCase):
    def test_extraction_answers_are_balanced(self) -> None:
        examples = RelationalRetrievalGenerator(seed=4).generate(
            extraction_size=80, transfer_per_example=0
        )
        self.assertEqual(set(answer_counts(examples).values()), {10})
        self.assertEqual(set(answer_counts(examples)), set(PAYLOADS))

    def test_answers_are_entailed_by_facts(self) -> None:
        examples = RelationalRetrievalGenerator(seed=9).generate(24, 3)
        for example in examples:
            facts = dict(example.facts)
            self.assertEqual(facts[example.query_name], example.answer)

    def test_first_two_transfer_variants_change_answer(self) -> None:
        examples = RelationalRetrievalGenerator(seed=2).generate(10, 2)
        by_group = {}
        for example in examples:
            by_group.setdefault(example.group_id, []).append(example)
        for group in by_group.values():
            base = next(example for example in group if example.split == "extraction")
            transfers = [example for example in group if example.split == "transfer"]
            self.assertTrue(all(item.answer != base.answer for item in transfers))

    def test_generation_is_reproducible(self) -> None:
        first = RelationalRetrievalGenerator(seed=7).generate(5, 2)
        second = RelationalRetrievalGenerator(seed=7).generate(5, 2)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

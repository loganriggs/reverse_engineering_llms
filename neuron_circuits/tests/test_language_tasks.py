import unittest

from neuron_circuits.language_tasks import (
    agreement_examples,
    bracket_examples,
    ioi_examples,
    language_counterfactual_pairs,
)


class LanguageTaskTests(unittest.TestCase):
    def test_ioi_answer_is_not_subject(self) -> None:
        for example in ioi_examples():
            self.assertNotEqual(example.answer_text, example.contrast_text)
            self.assertIn(example.answer_label, example.prompt)

    def test_agreement_is_balanced(self) -> None:
        examples = agreement_examples()
        self.assertEqual(
            sum(example.answer_text == " is" for example in examples),
            sum(example.answer_text == " are" for example in examples),
        )

    def test_bracket_answer_matches_unclosed_opener(self) -> None:
        expected = {"(": ")", "[": "]", "{": "}"}
        for example in bracket_examples():
            sequence = example.prompt.rsplit("\n", 1)[-1].split(" ->", 1)[0]
            self.assertEqual(example.answer_label, expected[sequence[0]])

    def test_counterfactual_pairs_change_answer_within_family(self) -> None:
        pairs = language_counterfactual_pairs()
        self.assertEqual(
            {pair.family for pair in pairs}, {"ioi", "agreement", "brackets"}
        )
        for pair in pairs:
            self.assertEqual(pair.base.family, pair.family)
            self.assertEqual(pair.source.family, pair.family)
            self.assertNotEqual(pair.base.answer_text, pair.source.answer_text)
            self.assertNotEqual(pair.base.prompt, pair.source.prompt)


if __name__ == "__main__":
    unittest.main()

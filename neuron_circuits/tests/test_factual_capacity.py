import unittest

import numpy as np
import torch

from neuron_circuits.factual_data import (
    ATTRIBUTES, VOCAB, build_manifest, encode_prompts, labels_for_rho,
    manifest_checks,
)
from neuron_circuits.factual_model import FactualTransformer, Intervention


class FactualDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_manifest(seed=7, replicas=1)

    def test_manifest_invariants(self):
        checks = manifest_checks(self.manifest)
        self.assertTrue(checks["unique_ids"])
        self.assertTrue(checks["structured_rules_match"])
        self.assertTrue(checks["full_table_balanced"])
        self.assertTrue(checks["splits_disjoint"])

    def test_nested_randomization_and_exact_balance(self):
        labels_half, selected_half, _ = labels_for_rho(self.manifest, 0.5)
        labels_full, selected_full, _ = labels_for_rho(self.manifest, 1.0)
        self.assertTrue(np.all(selected_half <= selected_full))
        self.assertTrue(np.array_equal(labels_half[selected_half], labels_full[selected_half]))
        for labels in (labels_half, labels_full):
            for attribute in range(ATTRIBUTES):
                self.assertEqual(np.ptp(np.bincount(labels[:, attribute], minlength=8)), 0)

    def test_templates_use_known_tokens_and_fixed_length(self):
        attributes = np.arange(8, dtype=np.uint8)
        entities = np.arange(8)
        for template in range(8):
            prompt = encode_prompts(self.manifest["ids"][entities], self.manifest["traits"][entities], attributes, template)
            self.assertEqual(prompt.shape, (8, 18))
            self.assertGreaterEqual(prompt.min(), 0)
            self.assertLess(prompt.max(), VOCAB.size)


class FactualModelTests(unittest.TestCase):
    def test_intervention_recomputes_downstream(self):
        torch.manual_seed(0)
        model = FactualTransformer(VOCAB.size, width=32, heads=1, layers=2)
        tokens = torch.randint(0, VOCAB.size, (4, 18))
        intact, intact_activations, _ = model(tokens, return_activations=True)
        changed, changed_activations, _ = model(
            tokens, intervention=Intervention(layer=0, channel=0), return_activations=True,
        )
        self.assertFalse(torch.equal(intact, changed))
        self.assertTrue(torch.equal(changed_activations[0][:, 0], torch.zeros(4)))
        self.assertFalse(torch.equal(intact_activations[1], changed_activations[1]))

    def test_all_one_channel_masks_preserve_logits(self):
        torch.manual_seed(1)
        model = FactualTransformer(VOCAB.size, width=32, heads=1, layers=2)
        tokens = torch.randint(0, VOCAB.size, (4, 18))
        intact = model(tokens)
        masked = model(tokens, channel_masks=[torch.ones(model.channels) for _ in model.blocks])
        self.assertTrue(torch.equal(intact, masked))


if __name__ == "__main__":
    unittest.main()

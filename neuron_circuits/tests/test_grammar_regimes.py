import numpy as np
import torch

from neuron_circuits.grammar_data import (
    HELDOUT_TEMPLATES, TRAIN_TEMPLATES, VOCAB, build_manifest, encode_prompts,
    encode_sequences, labels_for_rho, language_model_weights, manifest_checks,
    restrict_train_rank,
)
from neuron_circuits.grammar_model import ActivationPatch, GrammarTransformer


def test_manifest_balance_splits_and_nested_randomization():
    manifest = build_manifest(0, train_per_stratum=4, heldout_pair_per_stratum=2, heldout_entity_per_stratum=2)
    checks = manifest_checks(manifest)
    assert all(value for key, value in checks.items() if key != "manifest_hash")
    labels0, selected0, _ = labels_for_rho(manifest, "train", 0)
    labels5, selected5, _ = labels_for_rho(manifest, "train", .5)
    labels1, selected1, _ = labels_for_rho(manifest, "train", 1)
    assert not selected0.any() and selected1.all()
    assert np.all(selected0 <= selected5) and np.all(selected5 <= selected1)
    for labels in (labels0, labels5, labels1):
        facts = manifest["train_facts"]
        for relation in range(8):
            assert np.ptp(np.bincount(labels[facts[:, 2] == relation], minlength=8)) == 0


def test_rank_restriction_preserves_nested_facts_and_targets():
    full = build_manifest(3, train_per_stratum=8, heldout_pair_per_stratum=2,
                          heldout_entity_per_stratum=2)
    small = restrict_train_rank(full, 3)
    medium = restrict_train_rank(full, 6)
    small_keys = {tuple(row) for row in small["train_facts"].tolist()}
    medium_keys = {tuple(row) for row in medium["train_facts"].tolist()}
    assert small_keys < medium_keys
    full_targets = {
        tuple(fact): int(target)
        for fact, target in zip(full["train_facts"], full["train_random_target"])
    }
    for manifest in (small, medium):
        labels, selected, _ = labels_for_rho(manifest, "train", 1)
        assert selected.all()
        assert all(int(label) == full_targets[tuple(fact)]
                   for fact, label in zip(manifest["train_facts"], labels))
        assert manifest_checks(manifest)["full_balance"]
        assert manifest["parent_hash"] == full["hash"]


def test_templates_use_known_vocabulary_and_language_weights():
    manifest = build_manifest(1, train_per_stratum=2, heldout_pair_per_stratum=1, heldout_entity_per_stratum=1)
    facts = manifest["train_facts"][:16]
    labels, _, _ = labels_for_rho(manifest, "train", .5)
    for template in (*TRAIN_TEMPLATES, *HELDOUT_TEMPLATES):
        prompt = encode_prompts(manifest, facts, template)
        sequence = encode_sequences(manifest, facts, labels[:16], template)
        weights = language_model_weights(sequence)
        assert prompt.shape == (16, 13) and sequence.shape == (16, 14)
        assert sequence.max() < VOCAB.size
        assert np.all(weights[:, -1] == 4)
        assert np.any(weights == 0) and np.any(weights == 1)


def test_activation_patch_recomputes_downstream():
    torch.manual_seed(0)
    model = GrammarTransformer(VOCAB.size, width=24, heads=3, layers=2)
    tokens = torch.randint(0, VOCAB.size, (4, 13))
    intact, activations = model(tokens, return_activations=True)
    unchanged = model(tokens, patch=ActivationPatch(0, activations[0]))
    changed = model(tokens, patch=ActivationPatch(0, torch.zeros_like(activations[0])))
    assert torch.equal(intact, unchanged)
    assert not torch.equal(intact, changed)


def test_channel_masks_and_projectors():
    torch.manual_seed(4)
    model = GrammarTransformer(VOCAB.size, width=24, heads=3, layers=2)
    tokens = torch.randint(0, VOCAB.size, (3, 13))
    logits, full = model(tokens, return_full_activations=True)
    ones = {layer: torch.ones(model.channels) for layer in range(model.layers_count)}
    identity = {layer: torch.eye(model.channels) for layer in range(model.layers_count)}
    zeros = {layer: torch.zeros(model.channels) for layer in range(model.layers_count)}
    assert full[0].shape == (3, 13, model.channels)
    assert torch.equal(logits, model(tokens, channel_masks=ones))
    assert torch.equal(logits, model(tokens, channel_projectors=identity))
    assert not torch.equal(logits, model(tokens, channel_masks=zeros))
    batched_ones = {
        layer: torch.ones(len(tokens), model.channels) for layer in range(model.layers_count)
    }
    assert torch.equal(logits, model(tokens, channel_masks=batched_ones))


def test_full_activations_retain_gradients():
    torch.manual_seed(5)
    model = GrammarTransformer(VOCAB.size, width=24, heads=3, layers=2)
    tokens = torch.randint(0, VOCAB.size, (1, 13))
    logits, activations = model(tokens, return_full_activations=True)
    gradients = torch.autograd.grad(logits[0, -1, 0], activations)
    assert all(gradient is not None for gradient in gradients)
    assert gradients[0].shape == activations[0].shape


def test_bilinear_and_swiglu_contracts():
    tokens = torch.randint(0, VOCAB.size, (3, 13))
    for kind in ("swiglu", "bilinear"):
        model = GrammarTransformer(VOCAB.size, width=24, heads=3, layers=3, mlp_kind=kind)
        logits, activations = model(tokens, return_activations=True)
        assert logits.shape == (3, 13, VOCAB.size)
        assert len(activations) == 3 and activations[0].shape == (3, 48)

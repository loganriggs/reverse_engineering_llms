"""Deterministic entity--attribute data for the factual-capacity pilot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


K = 8
ATTRIBUTES = 8
TRAITS = 4
ID_DIGITS = 5
TEMPLATE_COUNT = 8
TRAIN_TEMPLATES = tuple(range(6))
HELDOUT_TEMPLATES = (6, 7)


@dataclass(frozen=True)
class Vocabulary:
    names: tuple[str, ...]
    bos: int
    answer_marker: int
    query_marker: int
    id_markers: tuple[int, int]
    trait_markers: tuple[tuple[int, ...], tuple[int, ...]]
    id_digits: tuple[int, ...]
    trait_values: tuple[tuple[int, ...], ...]
    attributes: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.names)


def build_vocabulary() -> Vocabulary:
    names: list[str] = []
    def add(name: str) -> int:
        names.append(name)
        return len(names) - 1
    bos = add("<bos>")
    answer_marker = add("<answer>")
    query_marker = add("<query>")
    id_markers = (add("<id>"), add("<entity>"))
    trait_markers = tuple(
        tuple(add(f"<z{trait + 1}{suffix}>") for trait in range(TRAITS))
        for suffix in ("", "_alt")
    )
    id_digits = tuple(add(f"i{value}") for value in range(K))
    trait_values = tuple(
        tuple(add(f"z{trait + 1}v{value}") for value in range(K))
        for trait in range(TRAITS)
    )
    attributes = tuple(add(f"a{attribute}") for attribute in range(ATTRIBUTES))
    return Vocabulary(tuple(names), bos, answer_marker, query_marker, id_markers, trait_markers, id_digits, trait_values, attributes)


VOCAB = build_vocabulary()


def structured_values(traits: np.ndarray) -> np.ndarray:
    """Return the eight unpermuted rule outputs for [entity, four traits]."""
    z1, z2, z3, z4 = traits.T
    return np.stack(
        (z1, z2, z3, z4, (z1 + z2) % K, (z3 + z4) % K,
         np.where(z3 < 4, z1, z2), np.where(z1 < 4, z3, z4)),
        axis=1,
    ).astype(np.uint8)


def _base8_digits(values: np.ndarray) -> np.ndarray:
    powers = K ** np.arange(ID_DIGITS - 1, -1, -1)
    return ((values[:, None] // powers[None, :]) % K).astype(np.uint8)


def _template_specs() -> tuple[tuple[int, tuple[int, ...]], ...]:
    # Six of the eight order/style combinations train the model.  Both marker
    # styles and every field order occur in training; templates 6 and 7 reserve
    # unseen combinations of familiar pieces.
    orders = (
        (0, 1, 2, 3, 4), (1, 2, 3, 4, 0),
        (0, 3, 1, 4, 2), (2, 0, 4, 1, 3),
    )
    return (
        (0, orders[0]), (1, orders[0]), (0, orders[1]), (1, orders[1]),
        (0, orders[2]), (1, orders[3]), (1, orders[2]), (0, orders[3]),
    )


def encode_prompts(
    ids: np.ndarray, traits: np.ndarray, attributes: np.ndarray, template: int,
) -> np.ndarray:
    """Encode batches of prompts. Field 0 is ID and fields 1..4 are traits."""
    style, order = _template_specs()[template]
    rows: list[np.ndarray] = [np.full((len(attributes), 1), VOCAB.bos, dtype=np.uint8)]
    for field in order:
        if field == 0:
            rows.append(np.full((len(attributes), 1), VOCAB.id_markers[style], dtype=np.uint8))
            rows.append(np.take(np.asarray(VOCAB.id_digits, dtype=np.uint8), ids))
        else:
            trait = field - 1
            rows.append(np.full((len(attributes), 1), VOCAB.trait_markers[style][trait], dtype=np.uint8))
            values = np.take(np.asarray(VOCAB.trait_values[trait], dtype=np.uint8), traits[:, trait])
            rows.append(values[:, None])
    rows.extend((
        np.full((len(attributes), 1), VOCAB.query_marker, dtype=np.uint8),
        np.take(np.asarray(VOCAB.attributes, dtype=np.uint8), attributes)[:, None],
        np.full((len(attributes), 1), VOCAB.answer_marker, dtype=np.uint8),
    ))
    encoded = np.concatenate(rows, axis=1)
    if encoded.shape[1] != 18:
        raise AssertionError(f"unexpected prompt length {encoded.shape[1]}")
    return encoded


def build_manifest(seed: int = 0, replicas: int = 4) -> dict[str, np.ndarray | int | str]:
    if replicas < 1 or replicas > 8:
        raise ValueError("replicas must be in [1, 8] for five base-8 ID digits")
    rng = np.random.default_rng(seed)
    grid = np.indices((K,) * TRAITS, dtype=np.uint8).reshape(TRAITS, -1).T
    traits = np.repeat(grid, replicas, axis=0)
    entity_count = len(traits)
    identity_values = rng.choice(K ** ID_DIGITS, size=entity_count, replace=False)
    ids = _base8_digits(identity_values)
    permutations = np.stack([rng.permutation(K) for _ in range(ATTRIBUTES)]).astype(np.uint8)
    rules = structured_values(traits)
    structured = permutations[np.arange(ATTRIBUTES)[None, :], rules].astype(np.uint8)

    # Each block contains one entity from every structured-answer stratum.  A
    # fixed answer permutation per block makes every rho prefix exactly balanced
    # and keeps a selected fact's random target stable across rho levels.
    block_rank = np.empty_like(structured, dtype=np.uint16)
    random_target = np.empty_like(structured, dtype=np.uint8)
    for attribute in range(ATTRIBUTES):
        strata = []
        for answer in range(K):
            members = np.flatnonzero(structured[:, attribute] == answer)
            rng.shuffle(members)
            strata.append(members)
        matrix = np.stack(strata, axis=1)
        for block in range(matrix.shape[0]):
            entities = matrix[block]
            targets = rng.permutation(K).astype(np.uint8)
            block_rank[entities, attribute] = block
            random_target[entities, attribute] = targets

    entity_order = rng.permutation(entity_count)
    heldout_entity_count = round(entity_count * 0.10)
    heldout_entities = np.zeros(entity_count, dtype=bool)
    heldout_entities[entity_order[:heldout_entity_count]] = True
    eligible_pairs = np.argwhere(~heldout_entities[:, None] & np.ones((1, ATTRIBUTES), dtype=bool))
    rng.shuffle(eligible_pairs)
    heldout_pair_count = round(len(eligible_pairs) * 0.10)
    heldout_pairs = np.zeros((entity_count, ATTRIBUTES), dtype=bool)
    heldout_pairs[tuple(eligible_pairs[:heldout_pair_count].T)] = True
    train_pairs = ~heldout_entities[:, None] & ~heldout_pairs

    payload = {
        "seed": seed, "replicas": replicas, "traits": traits, "ids": ids,
        "permutations": permutations, "rules": rules, "structured": structured,
        "block_rank": block_rank, "random_target": random_target,
        "heldout_entities": heldout_entities, "heldout_pairs": heldout_pairs,
        "train_pairs": train_pairs,
    }
    digest = hashlib.sha256()
    for key in sorted(payload):
        value = payload[key]
        digest.update(key.encode())
        digest.update(np.asarray(value).tobytes())
    payload["hash"] = digest.hexdigest()
    return payload


def labels_for_rho(manifest: dict, rho: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = int(np.asarray(manifest["block_rank"]).max()) + 1
    selected = np.asarray(manifest["block_rank"]) < round(rho * blocks)
    structured = np.asarray(manifest["structured"])
    labels = np.where(selected, np.asarray(manifest["random_target"]), structured).astype(np.uint8)
    changed = selected & (labels != structured)
    return labels, selected, changed


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in manifest.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(path, **arrays)
    metadata = {key: value for key, value in manifest.items() if not isinstance(value, np.ndarray)}
    metadata.update({"vocabulary": VOCAB.names, "train_templates": TRAIN_TEMPLATES, "heldout_templates": HELDOUT_TEMPLATES})
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path) as stored:
        return {**metadata, **{key: stored[key] for key in stored.files}}


def manifest_checks(manifest: dict) -> dict[str, bool | int | str]:
    traits = np.asarray(manifest["traits"])
    structured = np.asarray(manifest["structured"])
    rules = structured_values(traits)
    expected = np.asarray(manifest["permutations"])[np.arange(ATTRIBUTES)[None, :], rules]
    full_balance = all(np.array_equal(np.bincount(structured[:, a], minlength=K), np.full(K, len(traits) // K)) for a in range(ATTRIBUTES))
    unique_ids = len(np.unique(np.asarray(manifest["ids"]), axis=0)) == len(traits)
    disjoint = not np.any(np.asarray(manifest["train_pairs"]) & np.asarray(manifest["heldout_pairs"]))
    return {
        "entity_count": len(traits), "fact_count": int(structured.size),
        "unique_ids": unique_ids, "structured_rules_match": bool(np.array_equal(structured, expected)),
        "full_table_balanced": full_balance, "splits_disjoint": disjoint,
        "manifest_hash": str(manifest["hash"]),
    }

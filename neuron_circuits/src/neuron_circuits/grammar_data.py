"""Balanced compositional mini-language with fixed arbitrary exceptions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


K = 8
RELATIONS = 8
ENTITY_DIGITS = 3
ENTITY_COUNT = K ** ENTITY_DIGITS
TRAIN_TEMPLATES = tuple(range(6))
HELDOUT_TEMPLATES = (6, 7)


@dataclass(frozen=True)
class GrammarVocabulary:
    names: tuple[str, ...]
    bos: int
    sentence_markers: tuple[int, int]
    subject_markers: tuple[int, int]
    object_markers: tuple[int, int]
    query_markers: tuple[int, int]
    answer_markers: tuple[int, int]
    digits: tuple[int, ...]
    relations: tuple[int, ...]
    answers: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.names)


def build_vocabulary() -> GrammarVocabulary:
    names: list[str] = []

    def add(name: str) -> int:
        names.append(name)
        return len(names) - 1

    bos = add("<bos>")
    sentence_markers = (add("<mira>"), add("<tovan>"))
    subject_markers = (add("<agent>"), add("<speaker>"))
    object_markers = (add("<patient>"), add("<listener>"))
    query_markers = (add("<asks>"), add("<seeks>"))
    answer_markers = (add("<means>"), add("<yields>"))
    digits = tuple(add(f"syllable_{value}") for value in range(K))
    relations = tuple(add(f"relation_{value}") for value in range(RELATIONS))
    answers = tuple(add(f"answer_{value}") for value in range(K))
    return GrammarVocabulary(
        tuple(names), bos, sentence_markers, subject_markers, object_markers,
        query_markers, answer_markers, digits, relations, answers,
    )


VOCAB = build_vocabulary()


def entity_digits() -> np.ndarray:
    values = np.arange(ENTITY_COUNT)
    powers = K ** np.arange(ENTITY_DIGITS - 1, -1, -1)
    return ((values[:, None] // powers[None, :]) % K).astype(np.uint8)


def entity_traits(digits: np.ndarray) -> np.ndarray:
    # Calibration with nested modular traits caused otherwise successful
    # models to memorize the finite table instead of learning the grammar.
    # Keep the shared multi-token names but make their three latent traits
    # position-specific values; relation rules below still require role
    # binding and conditional composition.
    return digits.copy().astype(np.uint8)


def relation_values(subject_traits: np.ndarray, object_traits: np.ndarray, relations: np.ndarray) -> np.ndarray:
    s0, s1, s2 = subject_traits.T
    o0, o1, o2 = object_traits.T
    all_values = np.stack((
        s0,
        s1,
        o0,
        o1,
        np.where(s2 < 4, s0, o0),
        np.where(o2 < 4, s1, o1),
        np.where(s0 < 4, o2, s2),
        np.where(o0 < 4, s2, o2),
    ), axis=1).astype(np.uint8)
    return all_values[np.arange(len(relations)), relations]


def _sample_balanced_pairs(
    rng: np.random.Generator,
    traits: np.ndarray,
    eligible: np.ndarray,
    permutations: np.ndarray,
    per_stratum: int,
    used: set[tuple[int, int, int]],
) -> np.ndarray:
    rows = []
    subjects, objects = np.where(eligible)
    for relation in range(RELATIONS):
        values = relation_values(traits[subjects], traits[objects], np.full(len(subjects), relation))
        labels = permutations[relation, values]
        for answer in range(K):
            candidates = np.flatnonzero(labels == answer)
            rng.shuffle(candidates)
            selected = []
            for candidate in candidates:
                key = (int(subjects[candidate]), int(objects[candidate]), relation)
                if key not in used:
                    used.add(key); selected.append(key)
                    if len(selected) == per_stratum:
                        break
            if len(selected) != per_stratum:
                raise RuntimeError(f"insufficient pairs for relation={relation} answer={answer}")
            rows.extend(selected)
    result = np.asarray(rows, dtype=np.int32)
    rng.shuffle(result)
    return result


def _randomization_metadata(
    rng: np.random.Generator, facts: np.ndarray, structured: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rank = np.empty(len(facts), dtype=np.uint16)
    random_target = np.empty(len(facts), dtype=np.uint8)
    for relation in range(RELATIONS):
        strata = []
        for answer in range(K):
            members = np.flatnonzero((facts[:, 2] == relation) & (structured == answer))
            rng.shuffle(members); strata.append(members)
        length = min(map(len, strata))
        if not all(len(values) == length for values in strata):
            raise AssertionError("fact table is not exactly balanced")
        matrix = np.stack(strata, axis=1)
        for block in range(length):
            indices = matrix[block]
            rank[indices] = block
            random_target[indices] = rng.permutation(K)
    return rank, random_target


def build_manifest(
    seed: int = 0,
    train_per_stratum: int = 128,
    heldout_pair_per_stratum: int = 16,
    heldout_entity_per_stratum: int = 16,
) -> dict:
    rng = np.random.default_rng(seed)
    digits = entity_digits()
    traits = entity_traits(digits)
    permutations = np.stack([rng.permutation(K) for _ in range(RELATIONS)]).astype(np.uint8)
    heldout_entities = np.zeros(ENTITY_COUNT, dtype=bool)
    heldout_entities[rng.choice(ENTITY_COUNT, round(0.1 * ENTITY_COUNT), replace=False)] = True
    nonheldout = ~heldout_entities
    train_eligible = nonheldout[:, None] & nonheldout[None, :]
    heldout_entity_eligible = heldout_entities[:, None] | heldout_entities[None, :]
    used: set[tuple[int, int, int]] = set()
    train = _sample_balanced_pairs(rng, traits, train_eligible, permutations, train_per_stratum, used)
    heldout_pair = _sample_balanced_pairs(rng, traits, train_eligible, permutations, heldout_pair_per_stratum, used)
    heldout_entity = _sample_balanced_pairs(rng, traits, heldout_entity_eligible, permutations, heldout_entity_per_stratum, used)

    payload: dict[str, object] = {
        "seed": seed,
        "train_per_stratum": train_per_stratum,
        "heldout_pair_per_stratum": heldout_pair_per_stratum,
        "heldout_entity_per_stratum": heldout_entity_per_stratum,
        "digits": digits,
        "traits": traits,
        "permutations": permutations,
        "heldout_entities": heldout_entities,
    }
    for split, facts in (("train", train), ("heldout_pair", heldout_pair), ("heldout_entity", heldout_entity)):
        values = relation_values(traits[facts[:, 0]], traits[facts[:, 1]], facts[:, 2])
        structured = permutations[facts[:, 2], values].astype(np.uint8)
        rank, random_target = _randomization_metadata(rng, facts, structured)
        payload[f"{split}_facts"] = facts
        payload[f"{split}_structured"] = structured
        payload[f"{split}_rank"] = rank
        payload[f"{split}_random_target"] = random_target

    digest = hashlib.sha256()
    for key in sorted(payload):
        digest.update(key.encode()); digest.update(np.asarray(payload[key]).tobytes())
    payload["hash"] = digest.hexdigest()
    return payload


def labels_for_rho(manifest: dict, split: str, rho: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rank = np.asarray(manifest[f"{split}_rank"])
    blocks = int(rank.max()) + 1
    selected = rank < round(rho * blocks)
    structured = np.asarray(manifest[f"{split}_structured"])
    labels = np.where(selected, np.asarray(manifest[f"{split}_random_target"]), structured).astype(np.uint8)
    changed = selected & (labels != structured)
    return labels, selected, changed


def restrict_train_rank(manifest: dict, rank_limit: int) -> dict:
    """Return a nested training-table view with stable facts and labels.

    Each randomization rank contains one fact from every relation/structured-
    answer stratum. Prefixing ranks therefore changes load without resampling
    identities or arbitrary targets. Held-out splits remain unchanged.
    """
    rank = np.asarray(manifest["train_rank"])
    available = int(rank.max()) + 1
    if not 0 < rank_limit <= available:
        raise ValueError(f"rank_limit must be in [1, {available}], got {rank_limit}")
    keep = rank < rank_limit
    restricted = dict(manifest)
    for key in ("train_facts", "train_structured", "train_rank", "train_random_target"):
        restricted[key] = np.asarray(manifest[key])[keep].copy()
    restricted["parent_hash"] = manifest["hash"]
    restricted["train_rank_limit"] = int(rank_limit)
    restricted["train_per_stratum"] = int(rank_limit)
    restricted["hash"] = hashlib.sha256(
        f"{manifest['hash']}:train_rank_limit={rank_limit}".encode()
    ).hexdigest()
    return restricted


def _template_specs():
    # (style, order), where S/O/Q are complete fields.  Held-out templates 6
    # and 7 are new combinations of styles and orders seen during training.
    return (
        (0, "SOQ"), (1, "SOQ"), (0, "OSQ"), (1, "OSQ"),
        (0, "SQO"), (1, "OQS"), (1, "SQO"), (0, "OQS"),
    )


def encode_prompts(manifest: dict, facts: np.ndarray, template: int) -> np.ndarray:
    style, order = _template_specs()[template]
    digits = np.asarray(manifest["digits"])
    batch = len(facts)
    fields = {
        "S": np.concatenate((
            np.full((batch, 1), VOCAB.subject_markers[style], dtype=np.uint8),
            np.take(np.asarray(VOCAB.digits, dtype=np.uint8), digits[facts[:, 0]]),
        ), axis=1),
        "O": np.concatenate((
            np.full((batch, 1), VOCAB.object_markers[style], dtype=np.uint8),
            np.take(np.asarray(VOCAB.digits, dtype=np.uint8), digits[facts[:, 1]]),
        ), axis=1),
        "Q": np.concatenate((
            np.full((batch, 1), VOCAB.query_markers[style], dtype=np.uint8),
            np.take(np.asarray(VOCAB.relations, dtype=np.uint8), facts[:, 2])[:, None],
        ), axis=1),
    }
    pieces = [
        np.full((batch, 1), VOCAB.bos, dtype=np.uint8),
        np.full((batch, 1), VOCAB.sentence_markers[style], dtype=np.uint8),
        *(fields[field] for field in order),
        np.full((batch, 1), VOCAB.answer_markers[style], dtype=np.uint8),
    ]
    encoded = np.concatenate(pieces, axis=1)
    if encoded.shape[1] != 13:
        raise AssertionError(encoded.shape)
    return encoded


def encode_sequences(manifest: dict, facts: np.ndarray, labels: np.ndarray, template: int) -> np.ndarray:
    answers = np.take(np.asarray(VOCAB.answers, dtype=np.uint8), labels)[:, None]
    return np.concatenate((encode_prompts(manifest, facts, template), answers), axis=1)


def language_model_weights(sequences: np.ndarray, answer_weight: float = 4.0) -> np.ndarray:
    """Weights for next-token targets; unpredictable content tokens are zero."""
    targets = sequences[:, 1:]
    weights = np.ones_like(targets, dtype=np.float32)
    unpredictable = np.isin(targets, np.asarray((*VOCAB.digits, *VOCAB.relations)))
    weights[unpredictable] = 0.0
    weights[np.isin(targets, np.asarray(VOCAB.answers))] = answer_weight
    return weights


def manifest_checks(manifest: dict) -> dict:
    digits = np.asarray(manifest["digits"])
    checks: dict[str, object] = {
        "unique_entity_names": len(np.unique(digits, axis=0)) == ENTITY_COUNT,
        "trait_reconstruction": bool(np.array_equal(entity_traits(digits), manifest["traits"])),
        "split_disjoint": True,
        "full_balance": True,
        "heldout_entity_clean": True,
        "manifest_hash": manifest["hash"],
    }
    keys = []
    for split in ("train", "heldout_pair", "heldout_entity"):
        facts = np.asarray(manifest[f"{split}_facts"])
        keys.append({tuple(row) for row in facts.tolist()})
        structured = np.asarray(manifest[f"{split}_structured"])
        for relation in range(RELATIONS):
            counts = np.bincount(structured[facts[:, 2] == relation], minlength=K)
            checks["full_balance"] = bool(checks["full_balance"] and np.ptp(counts) == 0)
    checks["split_disjoint"] = not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    heldout = np.asarray(manifest["heldout_entities"])
    train = np.asarray(manifest["train_facts"]); entity = np.asarray(manifest["heldout_entity_facts"])
    checks["heldout_entity_clean"] = bool(
        not np.any(heldout[train[:, 0]] | heldout[train[:, 1]])
        and np.all(heldout[entity[:, 0]] | heldout[entity[:, 1]])
    )
    return checks


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

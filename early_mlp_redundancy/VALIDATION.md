# Validation performed on 2026-10-28

These checks validate the new inspection tool, not the historical scientific conclusions.

- Loaded the pinned checkpoint with SHA256 verification enabled using an explicit snapshot path.
- Loaded it again using the standard Hugging Face cache, from a working directory outside the packet.
- Two distinct prompts, 13 and 14 tokens; float32 on CUDA; PyTorch 2.14.1+cu130.
- Captured MLP0, MLP1 and MLP2 inputs, factors, products, and writes.
- Compared traced logits with the original GPT.forward → Block.forward execution, using a hook
  on the unembedding to obtain the native logits. Maximum discrepancy: exactly zero on both prompts.
- Identity write interventions reproduce traced logits exactly.
- Reconstructed each MLP write from its native Left/Right/Down/bias factors.
- Checked the full finite bilinear identity against direct evaluation at perturbed normalized inputs;
  maximum relative error across the first prompt's three layers was 3.27e-5 (threshold 1e-3).
- Ran individual and joint zero-write removals through the live suffix. JSON receipts retain KL,
  CE and top1 metrics; these small prompts are not a representative redundancy benchmark.
- Source files are hash-bound in PROVENANCE.json. Packet files are hash-bound in MANIFEST.json.

Dependencies were supplied by an existing local environment plus its isolated einops/tiktoken
dependency directory. No original research repository imports are needed by inspect_model.py.
A fresh pip installation, fresh network checkpoint download and CPU execution were not tested.
The lower-bound versions in requirements.txt express API requirements, not a tested version matrix.
The original archival scripts have not all been reexecuted and may need missing external data.

Follow-up: inspected historical composition source boundaries and verified simple arithmetic,
stored compensation gates, and compiler terminal authority. No new native forwards or full
bootstrap replay were performed for this amendment; see validation/history_audit.json.

EMLP-P1: 288 exploratory native-model forwards plus 3 replay forwards. See
NATIVE_MLP2_RESPONSE_PILOT.md and its separate result/audit receipts. This extends the
packet beyond the original loader smoke test; it does not validate corpus-level redundancy.

EMLP-P2: 704 forwards, exact identity/pre-edit replay, all stored overall summary means
checked, and 56 independent NumPy positionwise KL checks from saved logits (max7.43e-16).
See MLP2_TOKEN_TRANSFER.md; no representative-corpus or semantic-feature claim.

EMLP-P3:704native forwards; identity/pre-edit replay exact0; all overall summary means
and causal equalities checked;56independent NumPy KLchecks max8.29e-16.
See MLP1_RESPONSE.md for scope and limitations.

EMLP-P4:288native forwards; exact identity/oracle controls;224saved vector rows independently
recomputed with NumPy (maxrelative1.08e-15). Output-KL ratios checked from stored rows.
No additional audit forwards; see MLP1_MEDIATION.md.

EMLP-P5:704native forwards; identity/joint replay exact0;32independent scalar KL checks
max3.06e-16, previous P4 prefix arm within1.96e-7absolute. See PREFIX_RESPONSE_FACTORIAL.md.

EMLP-P6:320native forwards; identity/pre-source controls exact0;32prior full-response arms
replay exactly;28independent scalar KL checks max6.94e-16; source-prefix identities verified.
See RESPONSE_POSITIONS.md for the eight-source-context limitation and stopping rule.

EMLP-P7:1072native forwards;80training memberships/76single-token selections/cohort gates
checked;160texts exactly retokenize; independent saved KL/margins agree within8.89e-16/1.78e-15.
Selection metadata unchanged. See TRAINING_TASK_SCREEN.md for source-diversity limits.

P8 adds the [fixed preamble competence and transfer study](PREAMBLE_TRANSFER.md), with [audit receipt](validation/preamble_audit.json).

P9 completes the [direction-by-norm factorial](DIRECTION_NORM_FACTORIAL.md), including its lexical holdout protocol and [audit receipt](validation/direction_norm_audit.json).

P10 adds [lexical direction transfer](LEXICAL_DIRECTION_TRANSFER.md), with explicit word-pair exclusions and [audit receipt](validation/lexical_direction_audit.json).

User-definition correction: [literal removal versus replacement and steering](REMOVAL_REDUNDANCY.md). EMLP-D1 has a [separate deletion audit](validation/mlp1_group_deletion_audit.json); its outputs do not establish exact equivalence.

[D2 matched-norm controls](DELETION_NORM_CONTROL.md) separate amplitude-only perturbations from literal deletion, with [audit](validation/deletion_norm_control_audit.json).

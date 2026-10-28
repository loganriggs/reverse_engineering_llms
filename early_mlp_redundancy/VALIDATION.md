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

# h64 post-training evaluation preparation

The separate evaluator is implemented and its numerical/file-handling checks pass. It has NOT yet evaluated the running h64 scientific sweep, and the separate audit has now been implemented and tested on fixtures, but not run on the full scientific results.

coverage_evaluation.py computes empirical loss in bounded sample blocks, exact Gaussian source errors with verified Wick moments, readout adaptation and quadratic-feature matching. It never materializes N×P×4 linear-factor products. Zero candidate quadratics receive matching cosine0 by convention. Oracle readouts are fitted once per teacher/budget, not separately for every student.

evaluate_task_coverage_h64.py requires training_complete.json before scientific evaluation. That file alone means trained, not evaluated or audited. It verifies checkpoint/data hashes, regenerates all source blocks, evaluates every saved case, and writes each case atomically. Resume checks code/protocol versions plus checkpoint, source/teacher/adaptation-data and result-array hashes before accepting cached cases. Changing these inputs requires a distinct analysis rather than mixing old and new results.

## Verification

- d4/h2 and d6/h3: direct polynomial expansion independently checks feature values, blocked empirical loss and Gaussian population errors; largest population discrepancy6.83e-13. Under- and overdetermined adaptation fits agree with independent explicit SVD and coefficient-space errors.
- A real saved h64 smoke checkpoint agrees with independent Torch float64 forward evaluation on32inputs and520outputs. This check uses /tmp/task_coverage_h64_smoke, created by train_task_coverage_h64.py --smoke; the producer smoke is not scientific training evidence.
- A small full-pipeline fixture covers12saved cases and24learned transfer metrics, verified against polynomial coefficients. Initial and resume runs exit0; resumed results are byte-identical and skip completed case evaluation.
- Deliberately altered temporary checkpoint and adaptation files are rejected. No scientific artifact was modified by those negative tests.

## Next

After all60training fits finish, run evaluate_task_coverage_h64.py and then audit_task_coverage_h64.py; keep training, evaluated and audited completion states separate. Avoid heavy CPU analysis while the same pinned core is dispatching GPU work. Existing training session14051 must be polled, not duplicated. The broader research goal remains active.

## Independent audit preparation

coverage_audit.py independently reconstructs literal products, empirical loss in127-row blocks, quadratic-function cosine from monomial coefficients, and readout solutions with explicit SVD. The exact population calculation shares the previously verified Wick kernel; it is not a wholly separate symbolic implementation. The helper passes all12small fixture cases,24learned transfer metrics and a negative test with a deliberately wrong population error.

audit_task_coverage_h64.py validates complete condition coverage, saved data/code/checkpoint hashes, teacher and initial-model files, inactive weights, logged curve coverage and basic resource consistency before scientific metric checks. Its training-only mode passes all12real-size two-step smoke checkpoints. It does not replay all3,000optimizer steps or independently measure GPU memory. Full metric auditing is resumable with separate source/result hashes.

Actual transfer fits are counted as120learned plus10shared oracle fits for the planned full study. The240per-case metric entries repeat those oracle controls; they are not240independent fits. The scientific audit is still pending while training runs.

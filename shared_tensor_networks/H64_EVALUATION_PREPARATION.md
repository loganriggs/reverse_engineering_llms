# h64 post-training evaluation preparation

The separate evaluator is implemented and its numerical/file-handling checks pass. It has NOT yet evaluated the running h64 scientific sweep, and the separate scientific audit remains to be implemented.

coverage_evaluation.py computes empirical loss in bounded sample blocks, exact Gaussian source errors with verified Wick moments, readout adaptation and quadratic-feature matching. It never materializes N×P×4 linear-factor products. Zero candidate quadratics receive matching cosine0 by convention. Oracle readouts are fitted once per teacher/budget, not separately for every student.

evaluate_task_coverage_h64.py requires training_complete.json before scientific evaluation. That file alone means trained, not evaluated or audited. It verifies checkpoint/data hashes, regenerates all source blocks, evaluates every saved case, and writes each case atomically. Resume checks code/protocol versions plus checkpoint, source/teacher/adaptation-data and result-array hashes before accepting cached cases. Changing these inputs requires a distinct analysis rather than mixing old and new results.

## Verification

- d4/h2 and d6/h3: direct polynomial expansion independently checks feature values, blocked empirical loss and Gaussian population errors; largest population discrepancy6.83e-13. Under- and overdetermined adaptation fits agree with independent explicit SVD and coefficient-space errors.
- A real saved h64 smoke checkpoint agrees with independent Torch float64 forward evaluation on32inputs and520outputs. This check uses /tmp/task_coverage_h64_smoke, created by train_task_coverage_h64.py --smoke; the producer smoke is not scientific training evidence.
- A small full-pipeline fixture covers12saved cases and24learned transfer metrics, verified against polynomial coefficients. Initial and resume runs exit0; resumed results are byte-identical and skip completed case evaluation.
- Deliberately altered temporary checkpoint and adaptation files are rejected. No scientific artifact was modified by those negative tests.

## Next

Implement and test the independent h64 scientific audit while the existing producer runs. After all60training fits finish, run evaluation and then audit; keep training, evaluated and audited completion states separate. Avoid heavy CPU analysis while the same pinned core is dispatching GPU work. Existing training session14051 must be polled, not duplicated. The broader research goal remains active.

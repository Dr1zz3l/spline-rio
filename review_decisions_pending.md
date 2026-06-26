# Review items — status (after the 2026-06-26 pass)

Paper is temporarily at **13 pages** (re-trim to 12 deferred). Cheap fixes
#2,4,5,6,9,11,12,13,14,16 were applied earlier (commit 4e758d5).

## DONE this pass
- **#15** — justified the 400:1 gyro/accel ratio + added the Huber-δ sensitivity note.
- **#10b** — pulled summary per-segment KITTI RPE into §VI-B (live edge: racing
  converges to 1.0–1.7% beyond 20 m from 5–9% at 5 m; backflips ~5.6%).
- **#7** — stated we deliberately keep their STOCK gate/process-noise (we never
  re-tuned them; out-of-the-box coupling is the question).
- **#1** — magnetometer-noise robustness experiment. **POSITIVE RESULT** (no
  fallback to the cheap caveat needed): with a slowly-varying 2° heading drift +
  1.5° white noise, live orientation degrades by at most **0.3°** (fast 2.88→3.17°;
  slow 1.97→1.88°; backflips 6.35→6.34°). Written into §VII + a 199-word abstract
  clause. New solver knobs: `--set mag_heading_bias_deg`, `--set mag_heading_noise_deg`.

## STILL PENDING (deferred by you)

### #3 — NEES benchmark (⚠️ this one is a real correctness bug, still in the paper)
§VI-F reports the per-window **median** NEES (ori 1.2–2.6, vel 1.4–5.4) but compares
it to **3**. For χ²₃ the *mean* is 3 but the **median ≈ 2.37**. As written this is a
statistics error a methods reviewer catches immediately.
- **A (2-line fix, recommend doing before the supervisor sees it):** keep the median,
  change the benchmark to 2.37, re-read (ori conservative-to-calibrated about 2.37;
  vel straddles 2.37).
- **B (optional re-run):** report the masked **mean** vs 3 (recompute in `nees_eval.py`).
Say the word and I'll apply A (it's nearly free).

### #8 — why only EKF-RIO is ported (not Huang / Girod)
One sentence; honest reason is code availability. I can verify Huang's and Girod's
code availability (web/arXiv) and write the sentence, or you tell me.

### #17 — novelty-claim literature check (pre-submission)
Quick arXiv sweep before submission to back "backflips at 10 rad/s on a single-chip
sensor are unexplored." Defer to pre-submit; I can run it whenever.

### Page budget
Re-trim 13 → 12 when you're ready (the magnetometer paragraph + RPE summary + #15
are the new content; easy to claw back via the levers we used before).

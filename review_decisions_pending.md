# Review items needing your decision (10, 1, 3, 7, 8) + deferred (17)

These are the points from `claude_opus_review.md` I did **not** auto-apply, because each
needs either a decision from you, a re-run/experiment, or a fact only you know. The other
~11 "cheap" items (2, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16) are being applied directly.

Context constant for all of these: the paper is at the **hard 12-page limit**. Anything
that adds lines needs a compensating trim, which is the main friction on #10.

---

## #10 — Bring per-segment RPE back into the paper

**Reviewer's point.** For a dead-reckoning odometry system, per-segment KITTI-style RPE
(error vs segment length) is the more standard, more informative position metric than the
single start-anchored ATE%. We currently defer it *entirely* to the released code (one-line
pointer in Experimental Setup). A reviewer is likely to ask for it to be "pulled forward."

**My assessment.** Valid — and notably, this is the *same* reviewer-bait I flagged when we
removed the RPE figure for the page budget. So the reviewer independently confirms the
concern. The tension is purely page count: we deleted the RPE figure + its §VI-C subsection
+ the setup definition (~0.5 column) to get from 13→12 pages.

**Options.**
- **A. Hold the line** (repo pointer only, as now). Cost: free. Risk: a reviewer asks for it;
  we add it in revision. Defensible because we *do* report drift%, settled/live, and
  whole-traj ATE — RPE is complementary, not missing.
- **B. Summary numbers only** (no figure). One prose sentence in §VI: "Per-segment
  translational RPE converges to ~1.0% (slow) / ~2.2% (fast) beyond 20 m and is 3–8% at
  ≤10 m segments (radar-rate-limited); full curves in the released code." Cost: ~2 lines →
  needs a compensating ~2-line trim elsewhere (I can find one). This is the reviewer's
  explicit minimum ask.
- **C. Restore the full RPE figure + subsection.** Cost: ~0.5 column → would push to 13 pages;
  only viable if we cut something substantive. Not recommended at the 12pp limit.

**Recommendation: B.** Cheap insurance, satisfies the reviewer, keeps 12 pages.
**Need from you:** A, B, or C?

---

## #1 — Idealized heading: caveat the headline, and optionally a magnetometer-noise re-run

**Reviewer's point.** The abstract ("orientation RMSE … below 3°") and conclusion (2.0°/2.9°
racing) lead with heading-aided orientation, but that heading is **noise- and bias-free MoCap
yaw** simulated as a ~100 Hz magnetometer. §VII-B is honest about it; the headline is not. The
strongest defensible attitude number we have — roll/pitch parity with the EKF baselines,
heading-*independent* — is buried in Table V(b).

**My assessment.** Valid framing issue. Two tiers of fix:
- **Writing-only (the minimum, part of "cheap"):** foreground roll/pitch in the abstract and
  state yaw as heading-sensor-dependent. *I will do this as a cheap edit regardless* unless you
  object — it's the safe baseline. (Listed here only so you see the connection.)
- **Experiment (your call):** inject a realistic magnetometer model into ψ_ref — say 1–2°
  white heading noise + a slowly varying bias — and re-report how much orientation degrades.
  - If it degrades ~1°: turns a weakness into a **robustness result** (strong).
  - If it degrades more: better to find out now than in review.

**Cost of the experiment.** Small code change (add noise+bias to the heading-prior reference
in `validate_live_solver.py`) + re-run the 3 SW configs + update 2–3 numbers + one sentence.
~1–2 hours. No new data needed.

**Recommendation:** do the writing caveat now; **do the re-run before submission** — it's
high-value and cheap, and the result is good either way.
**Need from you:** want me to spec + run the magnetometer-noise experiment now, or just keep
the writing caveat for this pass?

---

## #3 — NEES is benchmarked against the wrong value (median vs mean)

**Reviewer's point.** §VI-F reports the per-window **median** NEES (orientation 1.2–2.6,
velocity 1.4–5.4) but calls **3** the "consistent value." For χ²₃ the *mean* is 3 but the
**median ≈ 2.37**. Comparing a median to 3 is a statistics error a methods reviewer catches
instantly.

**My assessment.** Correct and it's a genuine bug (I verified: median of χ²₃ ≈ 2.366). The
qualitative story survives, but the benchmark is wrong as written. Two fixes:
- **A. Writing-only (cheap):** keep the median, change the benchmark to **2.37** and re-read:
  orientation (1.2–2.6) is conservative-to-near-calibrated about 2.37; velocity (1.4–5.4)
  straddles 2.37 (calibrated-to-mildly-overconfident). ~2-line change, no re-run.
- **B. Re-run (cleaner):** report the **mean** NEES with a velocity-GT-quality mask (analogous
  to the backflips GT mask we already apply), compared to **3**. This is the statistically
  proper object and removes the finite-difference-MoCap-velocity outlier windows. Cost: a
  recompute in `nees_eval.py` (we have the per-window covariances saved) + update numbers.

**Recommendation:** **A now** (it's a correctness fix, must ship regardless), **B optionally
before submission** if you want the cleaner mean-vs-3 statement. I grouped this with the
decisions only because you paired "1 & 3"; if you want, I'll just apply A as a cheap edit.
**Need from you:** A only, or A now + B before submission?

---

## #7 — "Stock configurations" for the ported EKF baselines

**Reviewer's point.** We port ekf-rio/x-rio to our racing data with their **stock**
configuration and report 99.7% gate rejection. A skeptic — possibly Doer/Trommer as reviewers
— will say "you didn't even try to re-tune our gate/process-noise for your platform." This is
called the single most attackable claim in an otherwise bulletproof section.

**My assessment.** Valid and tactically important given the likely reviewer pool. The fix is
one sentence — but **which** sentence depends on a fact I don't have:
- **If you/we DID attempt a good-faith gate/Q adaptation and it still failed:** say so, and
  list the parameters tried. (Strongest rebuttal.)
- **If we ran stock deliberately** to characterize out-of-box coupling: say that explicitly,
  and why out-of-box is the relevant question for a "sensor/stack-coupling characterization,
  not a head-to-head" (which is already the section's framing).

**Need from you (the key fact):** did we ever try widening their innovation gate / inflating
process noise to admit our sparse, vibrating radar — or was it strictly stock? I will NOT
claim we tried re-tuning if we didn't. Tell me which is true and I'll write the matching
sentence.

---

## #8 — Why only the EKF-RIO family is ported (not Huang / Girod)

**Reviewer's point.** Huang et al. [12] (single-chip, discrete-time sliding-window
*optimization*) and Girod et al. [15] (single-chip multirotor smoother) are methodologically
*closer* to us than the EKF line, yet we port neither. "RMSE isn't comparable across papers"
explains why we don't *quote* them — not why we don't *port* them like we ported ekf-rio. A
reviewer will ask.

**My assessment.** Valid. The honest reason is almost certainly **code availability**: Doer &
Trommer's `ekf_rio`/`x_rio` toolbox is public (ROS), which is why a faithful port was
possible; Huang's and Girod's systems don't have comparably runnable releases. One sentence
fixes it — but I should state the true reason, not guess.

**Options.**
- I can **verify** code availability now (quick web/arXiv check for Girod BRIO and Huang
  "less is more" repos) and write the sentence accordingly, **or**
- you tell me directly (you may already know), and I'll write it.

**Need from you:** want me to verify their code availability, or do you already know whether
either released runnable code?

---

## #17 (deferred, not a paper edit) — novelty-claim literature check

"Backflips at 10 rad/s on a single-chip sensor are, to our knowledge, unexplored." Fine to
claim, but it's the kind a reviewer disproves with one citation. This is a **pre-submission**
task (your "before submit" bucket): a quick arXiv sweep of the last few months before you
submit. I can run that search now if you want a current read; otherwise it waits.

---

### Summary of what I need from you
1. **#10:** A (hold), B (summary numbers — recommended), or C (full figure)?
2. **#1:** magnetometer-noise re-run now, or writing caveat only this pass?
3. **#3:** A only (compare median to 2.37), or A now + B (masked mean) before submission?
4. **#7:** did we attempt re-tuning Doer's gate/Q, or strictly stock?
5. **#8:** should I verify Huang/Girod code availability, or do you know it?
6. **#17:** run the arXiv novelty check now, or defer?

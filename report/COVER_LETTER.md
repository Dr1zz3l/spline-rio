# Cover Letter — IEEE Transactions on Robotics

**Manuscript:** *Continuous-Time Radar-Inertial Odometry with B-Spline Sliding Window
Optimization*
**Authors:** Timo Weiss, Lukas Pries (Technical University of Munich)
**Type:** Regular Paper

---

Dear Editor-in-Chief,

We submit the above manuscript for consideration as a Regular Paper in the *IEEE
Transactions on Robotics*. The work presents a complete continuous-time radar-inertial
odometry (RIO) system for a quadrotor using a **single-chip 60 GHz mmWave radar** and an
IMU, and — to our knowledge — the **first evaluation of single-chip RIO through aggressive
racing and attitude-tumbling (backflip) flight up to 10 rad/s**, a regime in which the
radar measurement model itself degrades and which existing RIO evaluations do not cover.

The principal contributions are:

1. **A consistency-analyzed fixed-lag B-spline smoother.** A factor-set selection rule
   that exploits B-spline local support makes the Schur-complement marginal *exactly
   closed*, eliminating the regime-dependent prior-weight tuning that an inconsistent
   marginal requires. The mechanism is distinct from, and complementary to,
   first-estimates-Jacobian methods, and we contribute a reusable diagnostic symptom
   pattern by which the inconsistent variant can be recognized.

2. **One dynamics-adaptive measurement-weighting law** covering hover to 10 rad/s with a
   single configuration — each component derived from a measured failure diagnosis rather
   than tuned ad hoc — validated on **held-out flights never used for tuning**.

3. **A RANSAC ego-velocity front-end** that we cross-validate, in both directions, against
   the EKF-RIO family of Doer and Trommer on the public ICINS-2021 flights, closing an
   order-of-magnitude vertical-drift gap and reaching position parity with the baselines.

4. **Calibrated uncertainty and honest negatives.** The estimator's orientation output
   covariance is NEES-consistent on racing flights, supporting its use as a factor source;
   we further document a representation-level study that *disproves* our own
   adaptive-knot-density hypothesis, and show the apparent elevation bias under tumbling is
   a flight-regime error proxy, not a physical antenna constant.

We believe the work fits T-RO's standards for thoroughness and rigor: the system is
released with **code and datasets for independent, deterministic reproduction**
(github.com/Dr1zz3l/spline-rio), every hyperparameter is disclosed, the external
baselines are faithfully reproduced, and the limitations and negative results are reported
in full. A supplementary video accompanies the submission.

This manuscript is original, has not been published previously, and is not under
concurrent consideration at any other venue. The authors have no conflicts of interest to
declare.

We thank you and the reviewers for your time and consideration.

Sincerely,
Timo Weiss and Lukas Pries
Technical University of Munich (TUM)

---
*TODO before submitting: confirm corresponding-author email, add ORCID iDs, and (if the
flight video is included) reference it here.*

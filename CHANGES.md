# CHANGES — RIO paper review pass (2026-06-13)

Surgical verification + edit pass on Sec. VI-F (cross-validation) and Related Work.
No numbers were invented; every figure below traces to a command/file/URL.

---

## 1. Verification findings (Tasks 1–3)

### TASK 1 — BLOCKER: barometer claim → **REFUTED**

**Claim under test (old text):** "their filters absorb it with the onboard barometer —
a sensor our pipeline does not use" (Sec. VI-F) and "lacking an altitude sensor" (Conclusion).

**Method (definitive, option a):** built nothing new — used the existing `rio-baselines`
docker image and pinned rio toolbox. Ran ekf-yrio on ICINS `flight_1` twice, barometer
fusion ENABLED vs DISABLED, identical protocol, evaluated with our causal metric.

Command (host `radar-iwr6843-driver/`):
```
docker run --rm -v "$PWD":/workspace rio-baselines bash -lc '... \
  roslaunch /workspace/baselines/configs/icins_ekf-rio_rosbag.launch \
    pkg:=ekf_yrio rosbag:=flight_1.bag altimeter_update:={true,false} out_bag:=...'
.venv/bin/python3 baselines/adapters/eval_rio_output.py icins_flight_1 <bag> --filter-topic /rio
```
(`configs/icins_ekf-rio_rosbag.launch` made `altimeter_update` an overridable arg for this.)

**Toggle is real (source):** `baselines/rio/ekf_rio/src/ekf_rio_ros.cpp:265` —
`if (config_.altimeter_update)` guards `ekf_rio_filter_.updateAltimeter(...)` at line 271.
The launch's explicit `<param altimeter_update>` overrides the yaml default
(`ekf_yrio_default.yaml: altimeter_update: True`, `sigma_altimeter: 5.0`).

**Log evidence the toggle took effect** (`baselines/results/baro_test/log_{true,false}.txt`):
- `altimeter_update=true`:  `[EkfYRioRos]: Initialized baro h_0: 29.4824`, baro update active.
- `altimeter_update=false`: no baro init, **0** baro updates.

**Result (vertical drift, ICINS flight_1, start-anchored causal metric):**

| run pair | baro ON vertical | baro OFF vertical | Δ |
|---|---|---|---|
| pair 1 | 0.301 m (0.21%) | 0.300 m (0.21%) | 0.001 m |
| pair 2 (logged) | 0.234 m (0.16%) | 0.271 m (0.19%) | 0.037 m |

Disabling the barometer changes vertical drift by **≤ 0.04 m** — within the EKF's
run-to-run nondeterminism, and negligible against our **16.9 m** vertical drift on the
same flight. `sigma_altimeter = 5.0 m` makes the baro a near-negligible soft constraint.

**Conclusion: the barometer is NOT the mechanism.** The baselines obtain accurate vertical
estimates from radar + IMU alone. The old explanation (vertical Doppler bias "structurally
unobservable to a Doppler-plus-IMU-only estimator," "absorbed by the barometer") is
**contradicted** — baro-off IS Doppler-plus-IMU-only, and it is still accurate in z.

**Action taken:** removed the barometer/altitude-sensor mechanism from Sec. VI-F and the
Conclusion; replaced with the neutral measured fact + the ablation result; flagged the
cause as unresolved. **Did not** substitute a new mechanism. See OPEN ITEMS §3.

**Task 1c (our re-run config):** our Direction-B baseline re-runs had baro **ON**
(`configs/icins_ekf-rio_rosbag.launch:29` `altimeter_update value="true"`), on the original
ICINS bags which contain `/sensor_platform/baro`. (Given the ablation, this did not
materially affect the published Table V/VI baseline numbers.)

### TASK 2 — Huang et al. "Less is more" [14] → **was mischaracterized, corrected**

**Source:** arXiv:2402.02200 (HTML full text, https://arxiv.org/html/2402.02200v1).
**Finding:** it is a **sliding-window nonlinear least-squares optimization solved with
Ceres-Solver**, NOT an EKF/filter. Quotes: *"The system is based on a sliding window
approach that takes multiple frames to estimate the trajectory of RIO"*; *"We solve the
entire optimization problem … using the Ceres-Solver with derived Jacobians."*
Discrete-time (IMU pre-integration between frames). "Correspondence estimation" = RCS-bounded
nearest-neighbor point matching; "residual functions" = IMU pre-integration + Doppler +
point-to-point geometric residuals in the optimization. Evaluated on **ground-robot** and
**ColoRadar** (handheld, indoor/outdoor) data — not aerial, not tumbling.

**Action:** Related Work no longer calls it "an EKF estimator … in discrete-time filtering
form"; now "a discrete-time sliding-window optimization … on ground-robot and handheld
platforms rather than aerial tumbling flight." Kept the accurate parts (RCS + Doppler
physics, sparse single-chip cloud, discrete-time). Removed the wrong "aerial platforms"
framing.

### TASK 3 — ICINS IMU rate 409 Hz → **CONFIRMED**

**Source 1 (rosbag):** measured `/sensor_platform/imu` in
`baselines/datasets/icins2021/.../flight_datasets/flight_1.bag` = **409.5 Hz mean** over
3001 messages (rosbags reader). **Source 2:** Doer dataset page
(christopherdoer.github.io/datasets/icins_2021_radar_inertial_odometry) states 409 Hz.
No edit needed (model ADIS16448 and radar IWR6843AOP were already verified).

---

## 2. LaTeX edits (diffs)

`report/references.bib` — added x-rio (proper citation, from the Springer PDF the user
supplied: Gyroscopy and Navigation 12(4):329–339, 2021, DOI 10.1134/S2075108721040039):
```
+@article{doer2021xrio,
+  author = {Doer, Christopher and Trommer, Gert F.},
+  title  = {x-{RIO}: Radar Inertial Odometry with Multiple Radar Sensors and Yaw Aiding},
+  journal = {Gyroscopy and Navigation}, volume = {12}, number = {4},
+  pages = {329--339}, year = {2021}, doi = {10.1134/S2075108721040039},
+}
```

`report/IEEE-conference-template-062824.tex` — four hunks:
1. Related Work: Huang re-characterized (Task 2).
2. Sec. VI-F orientation prose: roll/pitch range `0.25`→`0.24` (Task 4a).
3. Sec. VI-F position prose: barometer/unobservability mechanism removed, replaced with
   measured fact + ablation result (Task 1). Reverse-port: added the ekf-yrio-vs-x-rio
   port-choice sentence with `\cite{doer2021xrio}` (Task 4b); "2.6 km / 2.8 km" softened to
   ">2 km, essentially unconstrained" (Task 4c).
4. Conclusion: "lacking an altitude sensor … vertical Doppler bias of their
   horizontal-boresight mount" → "carries a large vertical drift … whose cause is
   unresolved — we ruled out altitude aiding by ablation" (Task 1).

`baselines/configs/icins_ekf-rio_rosbag.launch` — made `altimeter_update` an overridable
arg (default `true`, unchanged behavior) so the ablation is reproducible.

(Full unified diff is in `git diff`; the PDF was rebuilt, 11 pages, builds clean, all
citations including `doer2021xrio` resolve.)

---

## 3. OPEN ITEMS — needs you, not an edit

### 3.1 (PROMINENT) The position result in Sec. VI-F no longer has a mechanism
The barometer explanation is refuted (§1 Task 1). The paper now honestly states:
our vertical drift on ICINS is large (16.9 m / flight_1), the baselines' is not
(~0.2–0.3 m), and **we do not know why**. This is a real gap a reviewer may press on.
What the ablation tells us: the baselines hold z with **radar + IMU only**, so the cause
is something in OUR pipeline, not their altitude aiding. Hypotheses to investigate
(NOT written into the paper — would be unverified):
  - a residual vertical ego-velocity bias in our WLS/solver on their radar geometry
    (16.9 m over ~186 s ≈ a steady ~0.09 m/s z-velocity bias);
  - our accel/gravity handling not pinning vertical velocity the way their EKF mechanization does;
  - our `radar_zbias` / elevation handling tuned for our pitched mount, mis-applied to their
    horizontal-boresight data.
**Decision needed:** investigate the mechanism (new analysis), or ship the honest "open"
framing as-is. I did NOT mark this resolved.

### 3.2 EKF baseline numbers have a ~0.04 m run-to-run nondeterminism floor
The two ablation pairs gave different absolutes (0.30 vs 0.23–0.27 m vertical) from the same
config — the rio playback EKF is not bit-deterministic (timing/threading). Negligible for
every claim in the paper, but the Table V/VI baseline figures carry this ~0.04 m noise.

### 3.3 Whole-platform vertical-bias narrative elsewhere
Sec. setup / FINDINGS still describe our own sensor's −0.5…−0.65 m/s elevation bias and the
27.5° pitched-mount pitch-calibration remedy (true for OUR racing bags). I did NOT touch
those — they are about our platform, not the ICINS cross-validation. But if 3.1 is
investigated, check consistency between that story and whatever explains the ICINS z drift.

---

## Status
Resolved: Task 2 (Huang), Task 3 (IMU rate), Task 4 (polish), x-rio citation.
Resolved-by-removal + escalated: Task 1 (barometer refuted; mechanism now open — §3.1).
**Not marked ready** — §3.1 is a substantive open question for you.

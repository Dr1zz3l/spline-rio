check if there is a radar config that also returns data when stationary
add quantization to forward and reverse model?


attitude ground truth, position fitted

---

check if the full solver.yaml params are printed into the validation plot or not. including the extrinsics at start and end

---

joint calibratoin of radar time offset and full extrinsics. maybe dynamic time-warping function to correct the USB timing jitter frame-by-frame. although, if i were to allow extrinsics and timing offset to be affected by the solver, in the moving window approach we would allow a "driftable" extrinsic/timing as the window moves, not one-off fixed value as in the global optimization

---


mocap yaw als weighted factor hinzufügen für ideal magnetormeter

---

allelerometer für putch roll https://madflight.com/AHRS/ mahony. aus dem fileter die measurement functions extracten wie sensor measurement zu state korreliert (was ist deren effekt), was ein fehler zwischen measurement und satte ist, der muss rausgezogen werden und als added cost term in state estimation hinzufgen.

goal is bounded attitude (if needed with mocap yaw as ideal magnetometer sim) and velocity errors. if possible with variance value, for slower and highly dynamic case.

---

sliding window with frames
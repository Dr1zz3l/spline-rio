check if there is a radar config that also returns data when stationary
add quantization to forward and reverse model?


attitude ground truth, position fitted

---

check if the full solver.yaml params are printed into the validation plot or not. including the extrinsics at start and end

---

joint calibratoin of radar time offset and full extrinsics. maybe dynamic time-warping function to correct the USB timing jitter frame-by-frame. although, if i were to allow extrinsics and timing offset to be affected by the solver, in the moving window approach we would allow a "driftable" extrinsic/timing as the window moves, not one-off fixed value as in the global optimization

---

check if gyro has different time offset to mocap than accelerometer

---

sliding window with frames

---

radar huber loss dependant on config
same as wraparound velocity

---

time step distance 0.3s window mit mocap vergleichen für error werte, nicht final optimized splie da future bias

---

test ohne radar, nur IMU. attitude error verhalten evaluieren auf slow racing. hyperparameter gleich lassen
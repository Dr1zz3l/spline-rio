check if there is a radar config that also returns data when stationary
add quantization to forward and reverse model?


attitude ground truth, position fitted

---

check if the full solver.yaml params are printed into the validation plot or not. including the extrinsics at start and end

---

joint calibratoin of radar time offset and full extrinsics. maybe dynamic time-warping function to correct the USB timing jitter frame-by-frame. although, if i were to allow extrinsics and timing offset to be affected by the solver, in the moving window approach we would allow a "driftable" extrinsic/timing as the window moves, not one-off fixed value as in the global optimization

---

investigate this error that occured
```bash
----------------------------------Iteration 7-----------------------------------
Orientation RMSE: 4.2 deg | delta max: 0.49° mean: 0.28°
Acc bias: [-0.061, -0.147, 0.089]  Gyr bias: [-0.001, -0.000, -0.000]
Computing analytical Jacobian...
Cost: total=2958.54 | radar=238.2 accel=354.4 gyro=2343.7 bnd_vel=20.6 bnd_pos=0.7 bnd_ori=0.1 bnd_acc=0.6 bnd_gyr=0.1 bias_prior=0.0 ext_prior=0.0
Jacobian: (42687, 3135), nnz=951290, sparsity=99.29%
  Accepted: cost 2958.5 -> 2958.4 (max |delta|=0.9°)
  Skipped re-linearization (max |delta|=0.9° < 1.0° threshold)
Update norm: 241564.961638
LM damping: 1.00e-10
Iteration time: 14.764s

----------------------------------Iteration 8-----------------------------------
Orientation RMSE: 4.4 deg | delta max: 0.90° mean: 0.50°
Acc bias: [-0.061, -0.147, 0.089]  Gyr bias: [-0.001, -0.000, -0.000]
Computing analytical Jacobian...
Cost: total=2958.39 | radar=238.2 accel=354.4 gyro=2343.8 bnd_vel=20.5 bnd_pos=0.7 bnd_ori=0.1 bnd_acc=0.6 bnd_gyr=0.1 bias_prior=0.0 ext_prior=0.0
Jacobian: (42687, 3135), nnz=951290, sparsity=99.29%
  Accepted: cost 2958.4 -> 2958.2 (max |delta|=1.2°)
  Re-linearized: absorbed max |delta|=1.2° into nominal
Update norm: 0.153583
LM damping: 1.00e-11
Iteration time: 25.305s

----------------------------------Iteration 9-----------------------------------
Orientation RMSE: 4.6 deg | delta max: 0.00° mean: 0.00°
Acc bias: [-0.061, -0.147, 0.089]  Gyr bias: [-0.001, -0.000, -0.000]
Computing analytical Jacobian...
Cost: total=2959.13 | radar=238.2 accel=354.4 gyro=2344.9 bnd_vel=20.3 bnd_pos=0.7 bnd_ori=0.0 bnd_acc=0.6 bnd_gyr=0.0 bias_prior=0.0 ext_prior=0.0
Jacobian: (42687, 3135), nnz=874602, sparsity=99.35%
/home/mouse/MyData/radar-iwr6843-driver/analysis/validate_nonlinear_solver.py:1316: MatrixRankWarning: Matrix is exactly singular
  delta_x = spsolve(H, -b)
Traceback (most recent call last):
  File "/home/mouse/MyData/radar-iwr6843-driver/analysis/validate_nonlinear_solver.py", line 2525, in <module>
    main()
  File "/home/mouse/MyData/radar-iwr6843-driver/analysis/validate_nonlinear_solver.py", line 2047, in main
    optimized_state = solve_trajectory_nonlinear(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mouse/MyData/radar-iwr6843-driver/analysis/validate_nonlinear_solver.py", line 1340, in solve_trajectory_nonlinear
    _, r_new = _build_jacobian(state, iteration_idx=iteration)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mouse/MyData/radar-iwr6843-driver/analysis/validate_nonlinear_solver.py", line 1201, in _build_jacobian
    return compute_jacobian_analytical(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mouse/MyData/radar-iwr6843-driver/analysis/validate_nonlinear_solver.py", line 648, in compute_jacobian_analytical
    k = round((v_pred - v_meas) / (2.0 * v_max))
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: cannot convert float NaN to integer
```
analysis/calibrate_extrinsics.py — standalone calibration script with:                                                                                                                      
   
  - so3_exp() — exponential map so(3) → SO(3) for rotation perturbations                                                                                                                      
  - load_bag_data() — loads bag with timing window, v_max, and yaw-flip flag from config
  - build_interpolators() — velocity/omega/SLERP interpolators from MoCap states                                                                                                              
  - collect_radar_points() — extracts flat arrays of (timestamp, position, v_meas) with range filtering                                                                                       
  - compute_predicted_dopplers_batch() — vectorised forward model via predict_doppler_velocity()                                                                                              
  - residual_fn_multibag() — closure for least_squares; handles per-bag yaw-flip, fixed-k Doppler unwrapping, and roll/yaw priors                                                             
  - calibrate_imu_timing() — gyro cross-correlation on a 200 Hz grid, per-axis then median                                                                                                    
  - run_calibration() — outer unwrap loop (max 3 iterations), scipy.optimize.least_squares with method='trf', loss='huber', covariance from (J^T J)^{-1} s², formatted report + suggested YAML
                                                                                                                                                                                              
  Usage:                                                                                                                                                                                      
  cd analysis/                                                                                                                                                                                
  python calibrate_extrinsics.py slow_racing_best_velocity  
  python calibrate_extrinsics.py slow_racing_best_velocity fast_racing_best_velocity                                                                                                          
  python calibrate_extrinsics.py fast_racing_best_velocity --optimize-translation                                                                                                             
  python calibrate_extrinsics.py slow_racing_best_velocity --full-rotation
# Visual-Inertial Odometry (VIO)

## Overview

This module implements a basic Visual-Inertial Odometry pipeline.

VIO combines:

- Camera motion estimation
- IMU inertial prediction

to estimate:

- position
- velocity
- orientation

without GPS.

---

# Components

## imu_preintegration.py

Performs IMU integration between camera frames.

Outputs:

- delta position
- delta velocity
- delta rotation

---

## reprojection.py

Computes reprojection error.

Used for:
- optimization
- visual correction
- tracking quality

---

## feature_track.py

Tracks image features across frames using optical flow.

---

## state_buffer.py

Stores recent navigation states.

Useful for:
- sliding window optimization
- bundle adjustment
- smoothing

---

## vio_estimator.py

Main VIO estimator.

Combines:
- IMU prediction
- visual correction

into one navigation estimate.

---

# Current Status

Implemented:
- IMU preintegration
- optical flow tracking
- reprojection error
- state prediction

Future upgrades:
- nonlinear optimization
- factor graph
- bundle adjustment
- loop closure
- tightly coupled VIO
- EKF/MSCKF
- GTSAM/Ceres integration

---

# Long-Term Goal

Reliable GPS-denied navigation for:

- UAV
- drones
- robotics
- autonomous systems
- PX4 integration

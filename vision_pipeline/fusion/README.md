# Fusion Module

This module performs multi-sensor fusion for GPS-denied navigation.

Current fusion sources:
- IMU
- Visual Odometry (VO)
- Altitude estimation
- Airspeed estimation
- Wind estimation

---

# Architecture

The fusion system follows a prediction/correction pipeline.

Prediction:
- IMU integration

Correction:
- Visual odometry
- Altitude measurements
- Airspeed measurements

---

# Files

## state.py

Defines the global navigation state:
- position
- velocity
- orientation
- altitude
- airspeed
- wind

---

## prediction.py

IMU prediction step.

Uses:
- gyro
- accelerometer

Outputs:
- predicted pose
- predicted velocity

---

## correction.py

Sensor correction step.

Uses:
- visual odometry
- altitude
- airspeed
- wind

---

## fusion_manager.py

Main interface for the entire fusion pipeline.

Combines:
- prediction
- correction

---

# Future Improvements

Planned upgrades:
- EKF fusion
- Quaternion-state propagation
- Bias estimation
- Magnetometer fusion
- Radar/LiDAR altitude fusion
- AI visual fusion
- Loop closure
- SLAM backend

---

# Current Status

Current implementation:
- modular
- lightweight
- suitable for research
- suitable for PX4 integration

Future versions will evolve toward:
- full VIO
- EKF navigation
- autonomous UAV navigation stack

# Altitude Estimation System

## Overview

This module implements the altitude estimation subsystem for the GPS-denied navigation stack.

The purpose of this subsystem is to estimate stable vertical position (Z-axis) and vertical motion for UAV navigation in environments where GPS may be unavailable or unreliable.

The system is designed with a modular architecture to support future integration with:
- PX4
- MAVLink
- EKF-based sensor fusion
- radar altimeters
- laser/LiDAR sensors
- barometers

---

# Objectives

The altitude subsystem provides:

- Vertical velocity estimation
- Altitude prediction from IMU data
- Altitude filtering and smoothing
- Foundation for future multi-sensor fusion

This module is part of a larger autonomous navigation pipeline including:
- Visual Odometry (VO)
- Visual-Inertial Odometry (VIO)
- Attitude estimation
- Air-data integration
- Full navigation state estimation

---

# Current Architecture

## Files

### `vertical_velocity.py`

Estimates vertical velocity using IMU acceleration.

Responsibilities:
- Integrate vertical acceleration
- Estimate vertical speed (vz)

---

### `altitude_filter.py`

Implements complementary filtering for altitude estimation.

Responsibilities:
- Fuse predicted altitude with measured altitude
- Reduce drift
- Smooth noisy measurements

---

### `altitude_estimator.py`

Main altitude estimation interface.

Responsibilities:
- Coordinate altitude pipeline
- Combine velocity and filtering
- Provide final altitude estimate

---

# Current Inputs

The current implementation uses:

- IMU vertical acceleration
- Simulated altitude measurements

Future versions will support:
- Barometer altitude
- Radar altimeter measurements
- Laser/LiDAR altitude
- Terrain-relative altitude

---

# Current Outputs

The subsystem currently estimates:

- Altitude (z)
- Vertical velocity (vz)

Future versions may estimate:
- Vertical acceleration bias
- Terrain height
- Landing state
- Hover stability metrics

---

# Sensor Fusion Strategy

The planned fusion architecture is:

IMU:
- Fast response
- High-frequency prediction
- Short-term accuracy

Barometer:
- Smooth altitude reference
- Long-term stability

Radar/Laser Altimeter:
- Precise near-ground altitude
- Landing support
- Terrain following

The long-term goal is integration into a full EKF-based navigation system.

---

# Future Improvements

Planned upgrades include:

- Extended Kalman Filter (EKF)
- Bias estimation
- Adaptive filtering
- Real PX4 sensor integration
- MAVLink support
- Terrain following
- Landing detection
- Altitude hold stabilization
- VO scale correction using altitude

---

# Engineering Philosophy

This subsystem is intentionally modular.

The goal is not only to create working scripts, but to build a scalable navigation architecture similar to real autonomous systems used in:
- PX4
- ArduPilot
- VINS-Fusion
- GPS-denied UAV navigation systems

The design prioritizes:
- readability
- modularity
- scalability
- sensor abstraction
- future hardware integration

---

# Project Status

Current status:

- Visual Odometry (VO): implemented
- IMU integration: implemented
- Orientation system: implemented
- Altitude subsystem: foundation implemented
- Full EKF fusion: planned

---

# Notes

This module currently operates as a software prototype and research platform.

Sensor models and algorithms will gradually evolve toward real-time embedded deployment and autonomous flight integration.

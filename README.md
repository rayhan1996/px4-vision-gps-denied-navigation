PX4 Vision-Based State Estimation (GPS-Denied)
🎯 Project Goal (Phase 1)

The main objective of this project is to estimate the drone's position, velocity, and orientation without relying on GPS.

Inputs: Camera frames + IMU measurements
Outputs: Pose (position & orientation) + velocity

In simple terms: the drone should know “where I am and how I moved” even when GPS is unavailable.
This is a fundamental Visual/Inertial Odometry module.

⚡ Phase 1 Scope
Build a software module (Autonomy / Estimation Layer) that provides accurate state estimates to the PX4 flight controller.
Capture time-synchronized frames from simulator, camera, or video stream.
Perform basic feature detection, tracking, and relative motion estimation.
Log data for analysis and drift evaluation.

This phase focuses on understanding the core estimation engine before integrating full SLAM or AI-based decision layers.

📦 Deliverables
Python/C++ module for frame capture & motion estimation
CSV logs of synchronized sensor data and frame timestamps
Visualizations of estimated trajectory
Ready for integration with PX4 SITL or real hardware

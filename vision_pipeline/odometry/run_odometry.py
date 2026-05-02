"""
Monocular Visual Odometry + IMU Prediction (EuRoC)

- ORB feature extraction
- Feature matching
- Motion estimation
- IMU prediction between frames
- Trajectory accumulation
"""

import cv2
import glob
import numpy as np
import os
import sys

# ===============================
# Fix import path
# ===============================
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
))

# ===============================
# Project imports
# ===============================
from feature_extractor import extract_features
from feature_matcher import match_features
from motion_estimator import estimate_motion, rotation_to_quaternion
from logger import ExperimentLogger
from trajectory import TrajectoryBuilder
from config import CAM0_PATH, K_EUROC

# ✅ IMU imports
from vision_pipeline.imu.imu_loader import load_imu_data, get_imu_segment
from vision_pipeline.imu.imu_integrator import integrate_imu


# ===============================
# Init Logger & Trajectory
# ===============================
logger = ExperimentLogger(experiment_name="mono_vo_euroc_imu")
trajectory = TrajectoryBuilder()

# ===============================
# Camera Intrinsics
# ===============================
K = np.array(K_EUROC, dtype=np.float64)

# ===============================
# Load images
# ===============================
image_paths = sorted(glob.glob(os.path.join(CAM0_PATH, "*.png")))
assert len(image_paths) > 1, "Not enough images"

print(f"✅ Loaded {len(image_paths)} images")

# ===============================
# 🔥 Load IMU data
# ===============================
imu_path = os.path.join(os.path.dirname(CAM0_PATH), "imu0", "data.csv")
imu_timestamps, gyro, accel = load_imu_data(imu_path)

# ===============================
# ⚠️ FAKE timestamps (temporary)
# ===============================
# Later we replace with real EuRoC timestamps
image_timestamps = np.linspace(0, len(image_paths) * 0.05, len(image_paths))

# ===============================
# IMU State Initialization
# ===============================
p = np.zeros(3)
v = np.zeros(3)
R_imu = np.eye(3)

prev_time = image_timestamps[0]

# ===============================
# First frame
# ===============================
prev_img = cv2.imread(image_paths[0], 0)
prev_kp, prev_desc = extract_features(prev_img)

# ===============================
# Main loop
# ===============================
for i, path in enumerate(image_paths[1:], start=1):

    curr_img = cv2.imread(path, 0)
    kp, desc = extract_features(curr_img)

    # ---------------------------
    # IMU PREDICTION
    # ---------------------------
    current_time = image_timestamps[i]

    ts, g, a = get_imu_segment(
        imu_timestamps, gyro, accel,
        prev_time, current_time
    )

    p, v, R_imu = integrate_imu(ts, g, a, p, v, R_imu)

    prev_time = current_time

    # ---------------------------
    # VISUAL ODOMETRY
    # ---------------------------
    if desc is None or prev_desc is None:
        prev_kp, prev_desc = kp, desc
        continue

    matches = match_features(prev_desc, desc)

    if len(matches) < 8:
        prev_kp, prev_desc = kp, desc
        continue

    R_vo, t_vo = estimate_motion(prev_kp, kp, matches, K)

    # ---------------------------
    # SIMPLE FUSION (TEMP)
    # ---------------------------
    # Use VO for direction, IMU helps stability
    R_imu = R_vo
    p = p + R_imu @ t_vo.flatten()

    # ---------------------------
    # Logging
    # ---------------------------
    q = rotation_to_quaternion(R_imu)

    trajectory.append(p)
    logger.log_pose(p, q)

    print(f"Frame {i}: Position {p}")

    prev_kp, prev_desc = kp, desc

# ===============================
# Save results
# ===============================
trajectory.save("trajectory.npy")
logger.save()

print("✅ Done.")

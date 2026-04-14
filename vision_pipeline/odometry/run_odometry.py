"""
Monocular Visual Odometry with Trajectory (EuRoC)

This script runs a full visual odometry pipeline:
- Feature extraction (ORB)
- Feature matching
- Motion estimation (R, t)
- Trajectory accumulation

Outputs:
- Logged poses
- Saved trajectory (trajectory.npy)
"""

import cv2
import glob
import numpy as np
import time
import os
import sys

# ===============================
# Fix import path (for config)
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

# ===============================
# Init Logger & Trajectory
# ===============================
logger = ExperimentLogger(experiment_name="mono_vo_euroc")
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
# First frame
# ===============================
prev_img = cv2.imread(image_paths[0], 0)
prev_kp, prev_desc = extract_features(prev_img)

# ===============================
# Main loop
# ===============================
for path in image_paths[1:]:

    curr_img = cv2.imread(path, 0)
    kp, desc = extract_features(curr_img)

    if desc is None or prev_desc is None:
        prev_kp, prev_desc = kp, desc
        continue

    matches = match_features(prev_desc, desc)

    if len(matches) < 8:
        prev_kp, prev_desc = kp, desc
        continue

    # ===============================

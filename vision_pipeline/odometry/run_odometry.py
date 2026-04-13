"""
Visual Odometry Runner (EuRoC MAV Dataset)

This script runs a monocular visual odometry pipeline using sequential images.
It performs:
- Feature extraction (ORB)
- Feature matching
- Relative motion estimation (R, t)
- Logging trajectory results

Designed to work with EuRoC MAV dataset (cam0 images).
"""

import cv2
import glob
import numpy as np
import time
import os

from feature_extractor import extract_features
from feature_matcher import match_features
from motion_estimator import estimate_motion, rotation_to_quaternion
from logger import ExperimentLogger

# ===============================
# Init Logger
# ===============================
logger = ExperimentLogger(experiment_name="orb_vo_euroc")

# ===============================
# EuRoC Camera Intrinsics (cam0)
# ===============================
K = np.array([
    [458.654, 0, 367.215],
    [0, 457.296, 248.375],
    [0, 0, 1]
])

# ===============================
# Dataset Path (CHANGE IF NEEDED)
# ===============================
DATASET_PATH = "/content/drive/MyDrive/EuRoC/MH_01_easy/mav0/cam0/data"

# ===============================
# Load images
# ===============================
image_paths = sorted(glob.glob(os.path.join(DATASET_PATH, "*.png")))

if len(image_paths) < 2:
    raise RuntimeError("❌ Not enough images for odometry")

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

    # Extract features
    kp, desc = extract_features(curr_img)

    # Skip if descriptors are invalid
    if desc is None or prev_desc is None:
        print("⚠️ Skipping frame due to missing descriptors")
        prev_kp, prev_desc = kp, desc
        continue

    # Match features
    matches = match_features(prev_desc, desc)

    if len(matches) < 8:
        print("⚠️ Not enough matches")
        prev_kp, prev_desc = kp, desc
        continue

    # Estimate motion
    R, t = estimate_motion(prev_kp, kp, matches, K)

    # Print debug
    print("Translation:", t.ravel())
    print("Rotation:\n", R)

    # ===============================
    # Use real timestamp from filename
    # ===============================
    try:
        timestamp = float(os.path.basename(path).replace(".png", "")) / 1e9
    except:
        timestamp = time.time()

    # Convert outputs
    t_vec = t.ravel()
    q = rotation_to_quaternion(R)

    # Log result
    logger.log(timestamp, t_vec, q)

    # Update previous frame
    prev_kp, prev_desc = kp, desc

# ===============================
# Cleanup
# ===============================
logger.close()

print("✅ Visual odometry finished.")

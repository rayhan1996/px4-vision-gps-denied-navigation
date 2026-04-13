"""
Monocular Visual Odometry (EuRoC)

This script performs visual odometry using a single camera (cam0).
It matches features between consecutive frames (temporal matching)
to estimate relative motion (R, t).

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
from config import CAM0_PATH, K_EUROC

# ===============================
# Init Logger
# ===============================
logger = ExperimentLogger(experiment_name="mono_vo_euroc")

# ===============================
# Camera Intrinsics (from config)
# ===============================
K = np.array(K_EUROC, dtype=np.float64)

# ===============================
# Load images (from config path)
# ===============================
image_paths = sorted(glob.glob(os.path.join(CAM0_PATH, "*.png")))

if len(image_paths) < 2:
    raise RuntimeError("❌ Not enough images in dataset path")

print(f"✅ Loaded {len(image_paths)} images from EuRoC")

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

    R, t = estimate_motion(prev_kp, kp, matches, K)

    print("t:", t.ravel())

    # EuRoC timestamp (from filename)
    try:
        timestamp = float(os.path.basename(path).replace(".png", "")) / 1e9
    except:
        timestamp = time.time()

    q = rotation_to_quaternion(R)
    logger.log(timestamp, t.ravel(), q)

    prev_kp, prev_desc = kp, desc

# ===============================
# Cleanup
# ===============================
logger.close()
print("✅ Monocular VO finished")

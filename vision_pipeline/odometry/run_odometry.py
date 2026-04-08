import cv2
import glob
import numpy as np
import time

from feature_extractor import extract_features
from feature_matcher import match_features
from motion_estimator import estimate_motion, rotation_to_quaternion
from logger import ExperimentLogger

# ===============================
# Init Logger
# ===============================
logger = ExperimentLogger(experiment_name="orb_vo")

# ===============================
# Camera Intrinsics (temporary)
# ===============================
K = np.array([
    [718.8560, 0, 607.1928],
    [0, 718.8560, 185.2157],
    [0, 0, 1]
])

# ===============================
# Load images
# ===============================
image_paths = sorted(glob.glob("../data/frames/*.png"))

if len(image_paths) < 2:
    raise RuntimeError("❌ Not enough images for odometry")

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
    matches = match_features(prev_desc, desc)

    # Estimate motion
    R, t = estimate_motion(prev_kp, kp, matches, K)

    # Print debug
    print("Translation:", t.ravel())
    print("Rotation:\n", R)

    # ===============================
    # Logging (IMPORTANT PART)
    # ===============================
    timestamp = time.time()

    t_vec = t.ravel()
    q = rotation_to_quaternion(R)

    logger.log(timestamp, t_vec, q)

    # Update previous
    prev_kp, prev_desc = kp, desc

# ===============================
# Cleanup
# ===============================
logger.close()

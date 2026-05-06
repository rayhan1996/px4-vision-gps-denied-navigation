import cv2
import glob
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from vision_pipeline.odometry.feature_extractor import extract_features
from vision_pipeline.odometry.feature_matcher import match_features
from vision_pipeline.odometry.motion_estimator import estimate_motion
from vision_pipeline.config import CAM0_PATH, K_EUROC

K = np.array(K_EUROC, dtype=np.float64)

image_paths = sorted(glob.glob(os.path.join(CAM0_PATH, "*.png")))
assert len(image_paths) > 1

p = np.zeros(3)
trajectory = []

prev_img = cv2.imread(image_paths[0], 0)
prev_kp, prev_desc = extract_features(prev_img)

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

    p = p + t.flatten()
    trajectory.append(p.copy())

    print("VO Position:", p)

    prev_kp, prev_desc = kp, desc

np.save("vo_trajectory.npy", np.array(trajectory))
print("✅ VO test done")

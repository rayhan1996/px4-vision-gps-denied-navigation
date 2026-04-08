import cv2
import glob
from feature_extractor import extract_features
from feature_matcher import match_features
from motion_estimator import estimate_motion
import numpy as np

# fake camera intrinsics (later replace with real)
K = np.array([
    [718.8560, 0, 607.1928],
    [0, 718.8560, 185.2157],
    [0, 0, 1]
])

image_paths = sorted(glob.glob("../data/frames/*.png"))

prev_img = cv2.imread(image_paths[0], 0)
prev_kp, prev_desc = extract_features(prev_img)

for path in image_paths[1:]:
    curr_img = cv2.imread(path, 0)

    kp, desc = extract_features(curr_img)
    matches = match_features(prev_desc, desc)

    R, t = estimate_motion(prev_kp, kp, matches, K)

    print("Translation:", t.ravel())
    print("Rotation:\n", R)

    prev_kp, prev_desc = kp, desc

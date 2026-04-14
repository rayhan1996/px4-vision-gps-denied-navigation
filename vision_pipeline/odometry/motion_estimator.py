"""
Enhanced Motion Estimation for EuRoC Dataset

This module estimates relative camera motion using epipolar geometry.
Improvements:
- Robust RANSAC filtering
- Inlier selection
- Numerical stability improvements
- Ready for VIO / stereo extension
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy


def estimate_motion(kp1, kp2, matches, K):

    if len(matches) < 8:
        raise ValueError("Not enough matches for motion estimation")

    # ===============================
    # Convert matches to points
    # ===============================
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # ===============================
    # Normalize points (important for stability)
    # ===============================
    pts1 = cv2.undistortPoints(
        np.expand_dims(pts1, axis=1), K, None
    ).reshape(-1, 2)

    pts2 = cv2.undistortPoints(
        np.expand_dims(pts2, axis=1), K, None
    ).reshape(-1, 2)

    # ===============================
    # Essential Matrix (robust)
    # ===============================
    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1e-3
    )

    if E is None:
        raise RuntimeError("Essential matrix estimation failed")

    # ===============================
    # Keep only inliers
    # ===============================
    inlier_pts1 = pts1[mask.ravel() == 1]
    inlier_pts2 = pts2[mask.ravel() == 1]

    # ===============================
    # Recover pose
    # ===============================
    _, R, t, _ = cv2.recoverPose(
        E,
        inlier_pts1,
        inlier_pts2
    )

    return R, t


def rotation_to_quaternion(R):
    r = R_scipy.from_matrix(R)
    return r.as_quat()  # [qx, qy, qz, qw]

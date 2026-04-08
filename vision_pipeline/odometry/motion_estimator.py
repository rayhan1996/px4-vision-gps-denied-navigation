import cv2
import numpy as np

def estimate_motion(kp1, kp2, matches, K):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    return R, t

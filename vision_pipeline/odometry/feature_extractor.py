import cv2

def extract_features(image):
    orb = cv2.ORB_create(2000)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

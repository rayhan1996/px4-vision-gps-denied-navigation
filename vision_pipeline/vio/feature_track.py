import cv2
import numpy as np


class FeatureTrackManager:
    """
    Track features across frames.
    """

    def __init__(self):

        self.previous_points = None

    def initialize(self, image):

        corners = cv2.goodFeaturesToTrack(
            image,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=10
        )

        if corners is not None:
            self.previous_points = corners

        return corners

    def track(
        self,
        prev_image,
        curr_image
    ):

        if self.previous_points is None:
            return None, None

        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_image,
            curr_image,
            self.previous_points,
            None
        )

        good_prev = self.previous_points[
            status.flatten() == 1
        ]

        good_curr = curr_points[
            status.flatten() == 1
        ]

        self.previous_points = good_curr.reshape(-1, 1, 2)

        return good_prev, good_curr

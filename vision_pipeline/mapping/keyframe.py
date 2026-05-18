import numpy as np


class Keyframe:
    """
    Important camera frame.
    """

    def __init__(
        self,
        frame_id,
        image,
        position,
        rotation
    ):

        self.frame_id = frame_id

        self.image = image

        self.position = np.array(position)

        self.rotation = rotation.copy()

        self.landmark_ids = []

    def add_landmark(self, landmark_id):

        self.landmark_ids.append(
            landmark_id
        )

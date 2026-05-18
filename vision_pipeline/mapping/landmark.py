import numpy as np


class Landmark:
    """
    3D map landmark.
    """

    def __init__(
        self,
        landmark_id,
        position
    ):

        self.id = landmark_id

        self.position = np.array(position)

        self.observations = 0

    def observe(self):

        self.observations += 1

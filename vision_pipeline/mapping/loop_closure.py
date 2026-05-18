import numpy as np


class LoopClosureDetector:
    """
    Detect revisited places.
    """

    def __init__(
        self,
        distance_threshold=2.0
    ):

        self.distance_threshold = (
            distance_threshold
        )

    def detect(
        self,
        current_position,
        previous_positions
    ):

        for i, pos in enumerate(previous_positions):

            dist = np.linalg.norm(
                current_position - pos
            )

            if dist < self.distance_threshold:

                return True, i

        return False, -1

import numpy as np


class WindEstimator:
    """
    Estimate wind vector.

    wind = ground_velocity - air_velocity
    """

    def __init__(self):

        self.wind = np.zeros(3)

    def update(
        self,
        ground_velocity,
        air_velocity_vector
    ):

        self.wind = (
            ground_velocity
            - air_velocity_vector
        )

        return self.wind

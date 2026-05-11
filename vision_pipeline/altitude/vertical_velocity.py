import numpy as np


class VerticalVelocityEstimator:
    """
    Estimate vertical velocity from vertical acceleration.
    """

    def __init__(self):
        self.vz = 0.0

    def update(self, az_world, dt):
        """
        Args:
            az_world: vertical acceleration in world frame (m/s^2)
            dt: timestep (seconds)

        Returns:
            vertical velocity (m/s)
        """

        self.vz += az_world * dt

        return self.vz

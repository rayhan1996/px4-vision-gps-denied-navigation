import numpy as np

from vision_pipeline.altitude.vertical_velocity import (
    VerticalVelocityEstimator
)

from vision_pipeline.altitude.altitude_filter import (
    AltitudeComplementaryFilter
)


class AltitudeEstimator:
    """
    Main altitude estimation system.
    """

    def __init__(self):

        self.vertical_velocity = VerticalVelocityEstimator()

        self.filter = AltitudeComplementaryFilter(alpha=0.98)

        self.altitude_imu = 0.0
        self.altitude = 0.0

    def update(
        self,
        az_world,
        dt,
        measured_altitude=0.0
    ):
        """
        Args:
            az_world: vertical acceleration in world frame
            dt: timestep
            measured_altitude: future barometer/radar altitude

        Returns:
            fused altitude estimate
        """

        # ---------------------------
        # Vertical velocity
        # ---------------------------
        vz = self.vertical_velocity.update(
            az_world,
            dt
        )

        # ---------------------------
        # IMU predicted altitude
        # ---------------------------
        self.altitude_imu += vz * dt

        # ---------------------------
        # Fuse with measured altitude
        # ---------------------------
        self.altitude = self.filter.update(
            self.altitude_imu,
            measured_altitude
        )

        return self.altitude

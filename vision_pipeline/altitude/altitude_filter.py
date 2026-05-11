class AltitudeComplementaryFilter:
    """
    Simple complementary filter for altitude fusion.

    Combines:
    - IMU predicted altitude
    - future barometer altitude
    """

    def __init__(self, alpha=0.98):

        self.alpha = alpha
        self.altitude = 0.0

    def update(self, imu_altitude, measured_altitude):

        self.altitude = (
            self.alpha * imu_altitude
            + (1.0 - self.alpha) * measured_altitude
        )

        return self.altitude

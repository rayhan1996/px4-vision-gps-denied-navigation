import numpy as np

from vision_pipeline.imu.imu_integrator import skew


class IMUPreintegrator:
    """
    IMU preintegration between two camera frames.
    """

    def __init__(self):

        self.reset()

    def reset(self):

        self.delta_p = np.zeros(3)
        self.delta_v = np.zeros(3)
        self.delta_R = np.eye(3)

        self.total_dt = 0.0

    def integrate(self, timestamps, gyro, accel):

        if len(timestamps) < 2:
            return

        gravity = np.array([0.0, 0.0, -9.81])

        for i in range(1, len(timestamps)):

            dt = timestamps[i] - timestamps[i - 1]

            if dt <= 0:
                continue

            omega = gyro[i]
            a = accel[i]

            # -----------------------------
            # Rotation update
            # -----------------------------
            dR = np.eye(3) + skew(omega * dt)

            self.delta_R = self.delta_R @ dR

            # Re-orthonormalization
            U, _, Vt = np.linalg.svd(self.delta_R)
            self.delta_R = U @ Vt

            # -----------------------------
            # Acceleration in world frame
            # -----------------------------
            a_world = self.delta_R @ a + gravity

            # -----------------------------
            # Velocity integration
            # -----------------------------
            self.delta_v += a_world * dt

            # -----------------------------
            # Position integration
            # -----------------------------
            self.delta_p += (
                self.delta_v * dt
                + 0.5 * a_world * dt**2
            )

            self.total_dt += dt

    def get_delta(self):

        return (
            self.delta_p,
            self.delta_v,
            self.delta_R,
            self.total_dt
        )

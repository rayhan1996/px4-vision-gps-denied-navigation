import numpy as np

from vision_pipeline.vio.imu_preintegration import (
    IMUPreintegrator
)

from vision_pipeline.vio.reprojection import (
    reprojection_error
)

from vision_pipeline.fusion.state import (
    NavigationState
)


class VIOEstimator:
    """
    Basic Visual-Inertial Odometry estimator.
    """

    def __init__(self):

        self.state = NavigationState()

        self.preintegrator = IMUPreintegrator()

    def predict(
        self,
        imu_timestamps,
        gyro,
        accel
    ):

        self.preintegrator.reset()

        self.preintegrator.integrate(
            imu_timestamps,
            gyro,
            accel
        )

        dp, dv, dR, dt = (
            self.preintegrator.get_delta()
        )

        # -----------------------------------
        # Predict state
        # -----------------------------------

        self.state.position += dp

        self.state.velocity += dv

        self.state.rotation = (
            self.state.rotation @ dR
        )

    def correct(
        self,
        observed_features,
        projected_features
    ):

        errors = []

        for obs, proj in zip(
            observed_features,
            projected_features
        ):

            err = reprojection_error(
                obs,
                proj
            )

            errors.append(err)

        mean_error = np.mean(errors)

        return mean_error

    def get_state(self):

        return self.state

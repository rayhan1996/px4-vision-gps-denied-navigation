import numpy as np

from vision_pipeline.imu.imu_integrator import (
    integrate_imu
)


class StatePredictor:
    """
    Prediction step using IMU.

    IMU performs:
    - orientation prediction
    - velocity prediction
    - position prediction
    """

    def predict(
        self,
        state,
        timestamps,
        gyro,
        accel
    ):

        p, v, R = integrate_imu(
            timestamps,
            gyro,
            accel,
            state.position,
            state.velocity,
            state.rotation
        )

        state.position = p
        state.velocity = v
        state.rotation = R

        return state

import numpy as np

from vision_pipeline.odometry.motion_estimator import (
    rotation_to_quaternion
)


class StateCorrector:
    """
    Correction step using:
    - visual odometry
    - altitude
    - airspeed
    """

    def correct_vo(
        self,
        state,
        R_vo,
        t_vo
    ):
        """
        Correct pose using visual odometry.
        """

        # Orientation correction
        state.rotation = R_vo

        # Position correction
        state.position = (
            state.position
            + state.rotation @ t_vo.flatten()
        )

        # Quaternion update
        state.quaternion = rotation_to_quaternion(
            state.rotation
        )

        return state

    def correct_altitude(
        self,
        state,
        altitude
    ):

        state.altitude = altitude

        return state

    def correct_airspeed(
        self,
        state,
        airspeed
    ):

        state.airspeed = airspeed

        return state

    def correct_wind(
        self,
        state,
        wind
    ):

        state.wind = wind

        return state

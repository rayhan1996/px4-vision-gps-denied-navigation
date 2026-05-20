import numpy as np


def rotation_to_quaternion(R):
    """
    Convert rotation matrix to quaternion.
    Returns [x, y, z, w]
    """

    q = np.zeros(4)

    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)

        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s

    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(
                1.0 + R[0, 0] - R[1, 1] - R[2, 2]
            )

            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s

        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(
                1.0 + R[1, 1] - R[0, 0] - R[2, 2]
            )

            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s

        else:
            s = 2.0 * np.sqrt(
                1.0 + R[2, 2] - R[0, 0] - R[1, 1]
            )

            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s

    return q


class StateCorrector:
    """
    Correction step for fusion.
    """

    def correct_vo(
        self,
        state,
        position_vo,
        rotation_vo
    ):
        """
        Correct pose using visual odometry.
        """

        # Replace orientation
        state.rotation = rotation_vo

        # Replace position with VO estimate
        state.position = position_vo

        # Update quaternion
        state.quaternion = (
            rotation_to_quaternion(
                state.rotation
            )
        )

        return state

    def correct_altitude(
        self,
        state,
        altitude
    ):
        """
        Correct altitude estimate.
        """

        state.altitude = float(
            altitude
        )

        return state

    def correct_airspeed(
        self,
        state,
        airspeed
    ):
        """
        Correct airspeed estimate.
        """

        state.airspeed = float(
            airspeed
        )

        return state

    def correct_wind(
        self,
        state,
        wind_vector
    ):
        """
        Correct wind estimate.
        """

        state.wind_vector = np.array(
            wind_vector,
            dtype=float
        )

        return state

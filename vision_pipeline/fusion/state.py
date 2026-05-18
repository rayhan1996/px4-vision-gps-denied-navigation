import numpy as np


class NavigationState:
    """
    Global navigation state.

    Stores:
    - position
    - velocity
    - orientation
    - altitude
    - airspeed
    - wind
    """

    def __init__(self):

        # Position (world frame)
        self.position = np.zeros(3)

        # Velocity (world frame)
        self.velocity = np.zeros(3)

        # Orientation rotation matrix
        self.rotation = np.eye(3)

        # Quaternion
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])

        # Altitude
        self.altitude = 0.0

        # Airspeed
        self.airspeed = 0.0

        # Wind vector
        self.wind = np.zeros(3)

        # Timestamp
        self.timestamp = 0.0

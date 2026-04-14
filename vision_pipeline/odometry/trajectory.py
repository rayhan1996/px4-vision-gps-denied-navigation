"""
Trajectory Builder

This module accumulates relative camera motion (R, t) to build
a global trajectory over time.

Each update applies a new transformation to the current pose.
"""

import numpy as np


class TrajectoryBuilder:
    def __init__(self):
        """
        Initialize trajectory with identity pose.
        """
        self.current_pose = np.eye(4)  # 4x4 identity matrix
        self.poses = [self.current_pose.copy()]

    def update(self, R, t):
        """
        Update trajectory using new rotation and translation.

        Args:
            R (np.ndarray): 3x3 rotation matrix
            t (np.ndarray): 3x1 translation vector
        """
        # Build transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()

        # Accumulate pose
        self.current_pose = self.current_pose @ T

        # Store pose
        self.poses.append(self.current_pose.copy())

    def get_positions(self):
        """
        Extract only translation components from poses.

        Returns:
            np.ndarray: Nx3 array of positions
        """
        return np.array([pose[:3, 3] for pose in self.poses])

    def reset(self):
        """
        Reset trajectory to initial state.
        """
        self.current_pose = np.eye(4)
        self.poses = [self.current_pose.copy()]
